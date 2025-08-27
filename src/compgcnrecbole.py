import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
from torch_scatter import scatter_add
import numpy as np


class CompGCNLayer(nn.Module):

    def __init__(self, in_dim, out_dim, act=F.relu, comp_op='sub', dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.comp_op = comp_op
        self.act = act

        self.dropout = nn.Dropout(dropout)
        self.W_dir = nn.ModuleDict({
            'in': nn.Linear(in_dim, out_dim, bias=False),
            'out': nn.Linear(in_dim, out_dim, bias=False),
            'loop': nn.Linear(in_dim, out_dim, bias=False)
        })
        self.W_rel = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)

        self.apply(xavier_normal_initialization)        

    def forward(self, ent_emb, rel_emb, edge_index, edge_type, edge_dir, edge_norm):
        src, dst = edge_index
        msg = self.compose(ent_emb[src], rel_emb[edge_type])
        out = torch.zeros_like(ent_emb)

        for dir_id, dir_name in enumerate(['out', 'in', 'loop']):
            mask = (edge_dir == dir_id)
            if mask.sum().item() == 0:
                continue
            W = self.W_dir[dir_name]
            norm = edge_norm[mask].unsqueeze(1)
            msg_dir = W(msg[mask]) * norm
            out = out.index_add(0, dst[mask], msg_dir)

        out = self.bn(out)
        out = self.act(self.dropout(out))
        rel_out = self.act(self.W_rel(rel_emb))

        return out, rel_out       

    def compose(self, ent, rel):
        if self.comp_op == 'sub':
            return ent - rel
        elif self.comp_op == 'mult':
            return ent * rel
        elif self.comp_op == 'add':
            return ent + rel
        elif self.comp_op == 'ccorr':
            return torch.fft.irfft(torch.fft.rfft(ent) * torch.conj(torch.fft.rfft(rel)), n=ent.shape[-1]) / ent.shape[-1]**0.5
        else:
            raise NotImplementedError(f"Composition operation {self.comp_op} not implemented")


class CompGCNRecBole(KnowledgeRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(CompGCNRecBole, self).__init__(config, dataset)

        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.margin = config['margin']
        self.alpha = config['alpha']
        self.comp_op = config['comp_op']
        self.num_bases = config['num_bases']
        self.l2_reg = config['l2_reg']

        self.entity_field = config['ENTITY_ID_FIELD']
        self.relation_field = config['RELATION_ID_FIELD']
        self.head_field = config['HEAD_ENTITY_ID_FIELD']
        self.tail_field = config['TAIL_ENTITY_ID_FIELD']

        self.n_entities = dataset.num(self.entity_field)
        self.n_relations = dataset.num(self.relation_field)
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        self.dummy_eid = self.n_entities
        self.n_entities += 1
    
        self._build_kg_edges(dataset)

        self._compute_edge_norm()

        self.printer = 0

        self.item_entity_ids = self._build_item_entity_ids(dataset).to(self.device)

        total_relations = 2 * self.n_relations + 1
        self.rel_basis = nn.Parameter(torch.randn(self.num_bases, self.embedding_size))
        self.rel_wt = nn.Parameter(torch.randn(total_relations, self.num_bases))

        self.user_emb = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_emb = nn.Embedding(self.n_entities, self.embedding_size)

        self.layers = nn.ModuleList([
            CompGCNLayer(self.embedding_size, self.embedding_size, comp_op=self.comp_op)
            for _ in range(self.n_layers)
        ])

        self.apply(xavier_normal_initialization)
        self.bpr_loss = BPRLoss()

    def _build_kg_edges(self, dataset):
        kg = dataset.kg_feat
        head = kg[self.head_field]
        rel = kg[self.relation_field]
        tail = kg[self.tail_field]

        edge_list = []
        
        # Original edges (out direction)
        edge_list.append((head, tail, rel, 'out'))
        
        # Inverse edges (in direction) - use relation ids from n_relations to 2*n_relations-1
        edge_list.append((tail, head, rel + self.n_relations, 'in'))
        
        # Self-loop edges - use relation id 2*n_relations
        loop_index = torch.arange(self.n_entities)
        loop_rel_id = torch.full_like(loop_index, 2 * self.n_relations)
        edge_list.append((loop_index, loop_index, loop_rel_id, 'loop'))

        # Direction mapping
        self.dir_map = {'out': 0, 'in': 1, 'loop': 2}
        
        # Concatenate all edges
        self.edge_index = torch.cat([torch.stack([src, dst]) for src, dst, _, _ in edge_list], dim=1)
        self.edge_type = torch.cat([r for _, _, r, _ in edge_list])
        self.edge_dir = torch.cat([
            torch.full_like(r, fill_value=self.dir_map[dir], dtype=torch.long) 
            for _, _, r, dir in edge_list
        ])

    def _compute_edge_norm(self):
        row, col = self.edge_index
        deg = scatter_add(torch.ones_like(col, dtype=torch.float), col, dim=0, dim_size=self.n_entities)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1), -0.5)
        self.edge_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    def _build_item_entity_ids(self, dataset):
        mapping = torch.full((self.n_items,), self.dummy_eid, dtype=torch.long)

        item_tok2id   = dataset.field2token_id[self.ITEM_ID]
        entity_tok2id = dataset.field2token_id[self.entity_field]

        for item_tok, entity_tok in dataset.item2entity.items():
            i_id = item_tok2id[item_tok]
            e_id = entity_tok2id[entity_tok]
            mapping[i_id] = e_id

        return mapping

    def get_rel_embed(self):
        coeff = F.softmax(self.rel_wt, dim=1)
        return torch.mm(coeff, F.normalize(self.rel_basis, dim=1))

    def compose(self, ent, rel):
        if self.comp_op == 'sub':
            return ent - rel
        elif self.comp_op == 'mult':
            return ent * rel
        elif self.comp_op == 'add':
            return ent + rel
        elif self.comp_op == 'ccorr':
            return torch.fft.irfft(torch.fft.rfft(ent) * torch.conj(torch.fft.rfft(rel)), n=ent.shape[-1]) / ent.shape[-1]**0.5
        else:
            raise NotImplementedError(f"Composition operation {self.comp_op} not implemented")

    def forward(self):
        ent = self.entity_emb.weight
        rel = self.get_rel_embed()

        device = ent.device
        edge_index = self.edge_index.to(device)
        edge_type = self.edge_type.to(device)
        edge_dir = self.edge_dir.to(device)
        edge_norm = self.edge_norm.to(device)
        
        for layer in self.layers:
            ent, rel = layer(ent, rel, edge_index, edge_type, edge_dir, edge_norm)

        # print(ent.norm(), rel.norm())
        # self._compute_edge_norm()

        return ent, rel

    def calculate_loss(self, interaction):
        ent_emb, rel_emb = self.forward()

        total_loss = 0

        if self.head_field in interaction:
            h = interaction[self.head_field]
            r = interaction[self.relation_field]
            t = interaction[self.tail_field]

            neg_tail_field = f'neg_{self.tail_field}'
            if neg_tail_field in interaction:
                neg_t = interaction[neg_tail_field]
                
                pos_dist = torch.norm(self.compose(ent_emb[h], rel_emb[r]) - ent_emb[t], dim=1, p=2)
                neg_dist = torch.norm(self.compose(ent_emb[h], rel_emb[r]) - ent_emb[neg_t], dim=1, p=2)
                kg_loss = torch.clamp(self.margin + pos_dist - neg_dist, min=0).mean()
                total_loss += kg_loss

        u = interaction[self.USER_ID]
        pos_i = interaction[self.ITEM_ID]
        neg_i = interaction[self.NEG_ITEM_ID]

        pos_scores = (self.user_emb(u) * ent_emb[self.item_entity_ids[pos_i]]).sum(-1)
        neg_scores = (self.user_emb(u) * ent_emb[self.item_entity_ids[neg_i]]).sum(-1)
        rec_loss = self.bpr_loss(pos_scores, neg_scores)
        total_loss += self.alpha * rec_loss

        l2 = (self.user_emb.weight.pow(2).sum() + ent_emb.pow(2).sum() + rel_emb.pow(2).sum())/ent_emb.size(0)
        total_loss += self.l2_reg * l2

        self.printer += 1

        if self.printer % 100 == 0:
            print("ent_emb norm:", ent_emb.norm().item())
            print("rel_emb norm:", rel_emb.norm().item())
            print("pos score", pos_scores.mean().item(), "neg score", neg_scores.mean().item())
            print("Total:", total_loss.item(), "KG:", kg_loss.item(), "Rec:", rec_loss.item())
            self.printer = 0

        return total_loss

    def predict(self, interaction):
        ent_emb, _ = self.forward()
        u_emb = self.user_emb(interaction[self.USER_ID])
        i_emb = ent_emb[self.item_entity_ids[interaction[self.ITEM_ID]]]
        return (u_emb * i_emb).sum(dim=1)

    def full_sort_predict(self, interaction):
        ent_emb, _ = self.forward()
        u = interaction[self.USER_ID]
        return torch.matmul(self.user_emb(u), ent_emb[self.item_entity_ids].t())

