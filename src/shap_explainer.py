import shap
import torch

def shap_explain(model, X, nsamples=100):
    """
    X: numpy array (N,D)
    model: pytorch model returning scalar logit
    """
    def f(x):
        tx = torch.from_numpy(x).float()
        with torch.no_grad():
            return model(tx).numpy()
    explainer = shap.KernelExplainer(f, X[:100])
    return explainer.shap_values(X, nsamples=nsamples)
