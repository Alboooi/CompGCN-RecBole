# CompGCN-RecBole — SIIA Exam Project — Multi‑Relational Recommendation with RecBole

## Overview

This repository contains the code and configuration assets for a course project in **Semantics in Intelligent Information Access (SIIA)**. The project investigates **multi‑relational recommendation** by integrating **knowledge‑graph embedding** and **user–item recommendation** within the **RecBole** framework. Two complementary families of models are implemented and evaluated:

* **TransERecBole** — a RecBole‑compatible adaptation of **TransE** for knowledge‑graph representation, trained jointly with a pairwise recommendation objective.
* **CompGCNRecBole** — a RecBole‑compatible adaptation of **Composition‑based Graph Convolutional Networks (CompGCN)** that propagates entity–relation messages and couples a KG loss with a recommendation loss.

The project emphasises *methodological clarity*, *reproducibility*, and *compatibility* with RecBole’s data pipeline and evaluation toolkit.

## Authorship and Context

**Author:** Alberto Ricchiuti
**Affiliation:** Department of Computer Science, University of Bari “Aldo Moro”
**Course:** Semantics in Intelligent Information Access (SIIA) — exam project.

## Key Features

* RecBole‑native **custom models**: `TransERecBole` and `CompGCNRecBole` under `src/`.
* **Ready‑to‑run** configuration files for both models in `configs/`.
* Minimal **runner scripts** in `to run/` that invoke RecBole’s `quick_start` API.
* Example **logs** and a brief **documentation** PDF under `docs/`.

## Repository Layout

```
siia-exam-main/
├── configs/
│   ├── config_TE.yaml          # TransERecBole configuration
│   └── config_CGCN.yaml        # CompGCNRecBole configuration
├── docs/
│   └── documentation.pdf       # Short project documentation/notes
├── src/
│   ├── __init__.py
│   ├── transerecbole.py        # TransERecBole model class
│   └── compgcnrecbole.py       # CompGCNRecBole model class
├── to run/
│   ├── run_TE.py               # Quick launcher for TransERecBole
│   ├── run_CGCN.py             # Quick launcher for CompGCNRecBole
│   └── runner.py               # Example RecBole quick_start usage
├── resulting logs/
│   └── logs.zip                # Sample logs (for reference)
└── README.md
```

## Prerequisites

* **Python** ≥ 3.9
* **PyTorch** (CPU or CUDA build, consistent with your environment)
* **RecBole** (latest stable release on https://github.com/RUCAIBox/RecBole.git)
* **pandas**
* For **CompGCN**: **torch‑scatter** compatible with your installed PyTorch/CUDA version

> **Note.** If you plan to run on GPU, ensure CUDA and the corresponding PyTorch/`torch-scatter` wheels are installed consistently.

### Suggested installation (conda + pip)

```bash
# Create an isolated environment (optional but recommended)
conda create -n siia python=3.10 -y
conda activate siia

# Install PyTorch matching your CUDA/CPU setup (visit pytorch.org for the exact command)
# Example (CPU‑only):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Core dependencies
pip install recbole pandas

# Required for CompGCN (choose a wheel matching your PyTorch/CUDA)
pip install torch-scatter
```

## Installing the Custom Models into RecBole (knowledge\_aware)

To enable RecBole to discover and instantiate the custom models in this repository, you must copy the model sources into RecBole’s **`knowledge_aware`** package and update its module initialiser.

1. **Locate RecBole’s `knowledge_aware` package** (inside your active Python environment):

   ```bash
   python - <<'PY'
   import pathlib, recbole.model.knowledge_aware as ka
   print(pathlib.Path(ka.__file__).parent)
   PY
   ```

2. **Copy the model files** from this repo into that folder:

   * `src/transerecbole.py`
   * `src/compgcnrecbole.py`

3. **Replace** `recbole/model/knowledge_aware/__init__.py` **with the version provided in this repository** (or, equivalently, **edit** your existing `__init__.py` to *expose* the new classes). Make sure it contains the following lines:

   ```python
   from .transerecbole import TransERecBole
   from .compgcnrecbole import CompGCNRecBole

   __all__ = [
       # ... existing models ...
       'TransERecBole',
       'CompGCNRecBole',
   ]
   ```

4. **Verify** that registration worked:

   ```bash
   python -c "from recbole.model.knowledge_aware import TransERecBole, CompGCNRecBole; print('OK')"
   ```

Once these steps are complete, the models can be invoked by **name** via RecBole’s `run_recbole` quick‑start API or via the provided runner scripts.

## Data

By default the configurations target **MovieLens‑100K** as provided in **RecBole’s `dataset_example/`**. You may either:

1. Point `data_path` in the YAML configs to RecBole’s example datasets inside your site‑packages, **or**
2. Download/prepare the dataset locally and point `data_path` to that folder.

> **Important.** Verify the following ID field names in your configs match the dataset schema:
> `USER_ID_FIELD`, `ITEM_ID_FIELD`, `ENTITY_ID_FIELD`, `HEAD_ENTITY_ID_FIELD`, `TAIL_ENTITY_ID_FIELD`, `RELATION_ID_FIELD`.

## Configuration

Two exemplar configuration files are provided:

* `configs/config_TE.yaml` — settings for **TransERecBole** (embedding dimension, margin for the translational loss, optimiser, sampling, evaluation `topk`, etc.).
* `configs/config_CGCN.yaml` — settings for **CompGCNRecBole** (graph convolution depth, composition operator, dropout, basis decomposition for relations if applicable, KG/Rec loss weights, optimiser, evaluation `topk`, etc.).

Common options include:

* **Training:** `epochs`, `train_batch_size`, negative sampling strategy.
* **Evaluation:** `valid_metric` (e.g., `Recall@10`), `topk`, `eval_batch_size`.
* **Reproducibility:** `seed`, `reproducibility`.
* **Hardware:** `use_gpu`, `device`, `gpu_id`.

> If your machine does not expose `gpu_id: 7` (as in the sample configs), change it to an available index (often `0`) or set `use_gpu: False` for CPU‑only runs.

## How to Use It

Follow this sequence for a clean, reproducible run.

**1) Create the environment**

* Install Python and core dependencies as in the *Prerequisites* section (PyTorch, RecBole, pandas, and—if using CompGCN—`torch-scatter` compatible with your PyTorch/CUDA).

**2) Register the custom models inside RecBole**

* Perform the steps in **Installing the Custom Models into RecBole (knowledge\_aware)** above (copy `.py` files, replace/edit `__init__.py`, verify imports).

**3) Prepare the dataset**

* Use RecBole’s provided MovieLens‑100K or point `data_path` in the YAML configs to your local dataset path. Ensure the ID field names in the configs match the data schema.

**4) Run with the supplied scripts (recommended)**
From the project root:

```bash
python "to run/run_TE.py"     # TransERecBole with configs/config_TE.yaml
python "to run/run_CGCN.py"   # CompGCNRecBole with configs/config_CGCN.yaml
```

These scripts invoke `recbole.quick_start.run_recbole` and write CSV summaries (`resuts_TE.csv`, `resuts_CGCN.csv`) with the returned metrics.

**5) Alternative: use RecBole’s quick\_start API directly**

```python
from recbole.quick_start import run_recbole

# TransE‑style model
result_te = run_recbole(
    model='TransERecBole',
    config_file_list=['configs/config_TE.yaml']
)

# CompGCN‑style model
result_cgcn = run_recbole(
    model='CompGCNRecBole',
    config_file_list=['configs/config_CGCN.yaml']
)
```

**6) Inspect outputs**

* Logs and console output are handled by RecBole; CSV summaries are saved by the runner scripts. Enable model checkpointing in the YAML if you wish to persist trained weights.

## Notes on the Custom Models

* **TransERecBole** couples a **pairwise recommendation loss** (BPR) with a **translational margin loss** on KG triples. It maps items to entity IDs to ensure user–item interactions and KG constraints co‑shape the embedding space.
* **CompGCNRecBole** performs **message passing** over the knowledge graph with relation‑aware composition operators (e.g., subtraction, multiplication, circular correlation), typically adding inverse relations and a self‑loop. The learned entity/user representations are optimised jointly with a recommendation objective.

## Reproducibility and Logging

* Seeds and deterministic flags are enabled in the configs to promote run‑to‑run stability.
* Example logs are included in `resulting logs/logs.zip` for reference.
* TensorBoard and/or Weights & Biases can be toggled via config flags if desired.

## Documentation

A short technical note is available at `docs/documentation.pdf` and summarises the modelling choices and the experimental protocol.

## Expected Outputs

* Console and log files managed by RecBole.
* CSV summaries written by the launcher scripts (`resuts_TE.csv`, `resuts_CGCN.csv`).
* (Optional) Saved checkpoints if `save_model: True` is retained in the configs.

## Acknowledgements

This work builds upon the **RecBole** library and foundational research on **TransE** and **CompGCN**. The author thanks colleagues and instructors of the SIIA course at the University of Bari for feedback and guidance.

## License and Usage

No explicit license file is provided in this repository. All rights reserved by the author unless otherwise specified. For academic use or collaboration enquiries, please contact the author.

## Citation

If you build upon this work in academic outputs, please cite the repository and course context. A suggested generic citation is:

> Ricchiuti, A. (2025). *Multi‑Relational Recommendation with RecBole (TransE & CompGCN)*. Course project for **Semantics in Intelligent Information Access**, Department of Computer Science, University of Bari “Aldo Moro”.

---

For clarifications or requests (e.g., access to trained checkpoints referenced in `models/links.txt`), please reach out to the repository owner.
