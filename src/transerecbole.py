import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class TransERecBole(KnowledgeRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.entity_field  = config["ENTITY_ID_FIELD"]
        self.relation_field = config["RELATION_ID_FIELD"]
        self.head_field     = config["HEAD_ENTITY_ID_FIELD"]
        self.tail_field     = config["TAIL_ENTITY_ID_FIELD"]

        self.n_entities = dataset.num(self.entity_field)
        self.n_relations = dataset.num(self.relation_field)
        self.n_users    = dataset.num(self.USER_ID)
        self.n_items    = dataset.num(self.ITEM_ID)

        self.ent_offset = self.n_users 
        self.dummy_eid  = self.n_entities + self.ent_offset
        self.n_entities += 1
        self.item_entity_ids = self._build_item_entity_ids(dataset).to(self.device)

        self.embedding_size = config["embedding_size"]
        self.margin = config["margin"]
        self.alpha = config["alpha"]
        self.l2_reg = config["l2_reg"]

        self.entity_embedding = nn.Embedding(self.n_users + self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.embedding_size)
        self.apply(xavier_normal_initialization)

        self.bpr_loss = BPRLoss()

    def _build_item_entity_ids(self, dataset):
        mapping = torch.full((self.n_items,), self.dummy_eid, dtype=torch.long)
        item_tok2id   = dataset.field2token_id[self.ITEM_ID]
        entity_tok2id = dataset.field2token_id[self.entity_field]

        for item_tok, ent_tok in dataset.item2entity.items():
            i_id = item_tok2id[item_tok]
            e_id = entity_tok2id[ent_tok] + self.ent_offset
            mapping[i_id] = e_id

        return mapping

    def kg_score(self, h, r, t):
        h_e = self.entity_embedding(h)
        r_e = self.relation_embedding(r)
        t_e = self.entity_embedding(t)
        return -(h_e + r_e - t_e).norm(p=2, dim=1)

    def rec_score(self, user, item_eid):
        u_e = self.entity_embedding(user)
        i_e = self.entity_embedding(item_eid)
        return (u_e * i_e).sum(-1)

    def calculate_loss(self, interaction):
        users = interaction[self.USER_ID]
        pos_items = interaction[self.ITEM_ID]
        neg_items = interaction[self.NEG_ITEM_ID]

        pos_eid = self.item_entity_ids[pos_items]
        neg_eid = self.item_entity_ids[neg_items]

        rec_loss = self.bpr_loss(self.rec_score(users, pos_eid), self.rec_score(users, neg_eid))

        kg_loss = torch.zeros(1, device=self.device)
        if all(k in interaction for k in (self.head_field, self.tail_field, self.relation_field)):
            ph = interaction[self.head_field] + self.ent_offset
            pt = interaction[self.tail_field] + self.ent_offset
            r  = interaction[self.relation_field]

            pos_kg = self.kg_score(ph, r, pt)

            neg_tail = f"neg_{self.tail_field}"
            if neg_tail in interaction:
                nt = interaction[neg_tail] + self.ent_offset
                neg_kg = self.kg_score(ph, r, nt)
                kg_loss = F.relu(self.margin - pos_kg + neg_kg).mean()
            else:
                kg_loss = pos_kg.mean()
        else:
            kg_loss = torch.zeros([], device=self.device)

        l2 = (self.entity_embedding.weight.pow(2).sum() + self.relation_embedding.weight.pow(2).sum())/self.entity_embedding.weight.size(0)

        return kg_loss + self.alpha * rec_loss + self.l2_reg * l2

    def predict(self, interaction):
        users = interaction[self.USER_ID]
        items = interaction[self.ITEM_ID]
        item_eid = self.item_entity_ids[items].to(self.device)
        return self.rec_score(users, item_eid)

    def full_sort_predict(self, interaction):
        users = interaction[self.USER_ID]
        item_e = self.item_entity_ids.to(self.device)
        return self.rec_score(users, item_e)
        
