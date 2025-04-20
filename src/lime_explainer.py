import lime
import lime.lime_tabular
import torch
import numpy as np

def lime_explain(model, X, feature_names):
    """
    X: numpy (N,D), feature_names: list of length D
    """
    def f(x):
        tx = torch.from_numpy(x).float()
        with torch.no_grad():
            return model(tx).numpy()
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X, feature_names=feature_names, verbose=False, mode='regression'
    )
    all_exp = []
    for i in range(X.shape[0]):
        exp = explainer.explain_instance(
            X[i], f, num_features=X.shape[1]
        ).as_list()
        # convert list-of-tuples → dense array
        arr = np.zeros(X.shape[1])
        for name, val in exp:
            idx = feature_names.index(name)
            arr[idx] = val
        all_exp.append(arr)
    return np.vstack(all_exp)
