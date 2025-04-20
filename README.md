# Neural Network Transparency & XAI for Credit Evaluation

This repository implements **TraceX**, a deterministic finite‑difference explainability method, and compares it with SHAP and LIME on a loan‑approval dataset.

## Structure

- `data/loan_data.csv`  
- `src/models.py` — DL architectures  
- `src/tracex.py` — TraceX implementation  
- `src/shap_explainer.py` — SHAP wrapper  
- `src/lime_explainer.py` — LIME wrapper  
- `src/train.py` — training, evaluation, and explainers  

## Quickstart

```bash
git clone TraceX
cd tracex
pip install -r requirements.txt

# Place your Kaggle CSV in data/loan_data.csv
python src/train.py \
    --model mlp \
    --explainer tracex \
    --output_dir outputs/mlp_tracex
