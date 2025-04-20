import argparse, os
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from models import MLP, CNN, TransformerModel, AutoencoderClassifier
from tracex import tracex_output, tracex_layerwise
from shap_explainer import shap_explain
from lime_explainer import lime_explain

def load_data(path):
    df = pd.read_csv(path)
    # assume last column 'target'; split and preprocess
    y = df['target'].values
    X_num = df.select_dtypes(include=['int','float']).drop('target',axis=1)
    X_cat = df.select_dtypes(include=['object'])
    # one‑hot encode
    enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
    Xc = enc.fit_transform(X_cat)
    # standardize
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X_num)
    X = np.hstack([Xn, Xc])
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(name, input_dim):
    if name=='mlp':          return MLP(input_dim)
    if name=='cnn':          return CNN(input_dim)
    if name=='transformer':  return TransformerModel(input_dim)
    if name=='autoencoder':  return AutoencoderClassifier(input_dim)

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['mlp','cnn','transformer','autoencoder'], required=True)
    p.add_argument('--explainer', choices=['tracex','shap','lime'], required=True)
    p.add_argument('--data', default='data/loan_data.csv')
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data(args.data)
    input_dim = X_train.shape[1]

    # train
    model = build_model(args.model, input_dim)
    criterion = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        model.train()
        logits = model(torch.from_numpy(X_train).float())
        loss = criterion(logits, torch.from_numpy(y_train).float())
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    # choose first 100 test points for explainers
    X0 = X_test[:100]

    if args.explainer=='tracex':
        # output‑level
        phis = [ tracex_output(model, torch.from_numpy(x).float()) for x in X0 ]
        np.save(os.path.join(args.output_dir,'phis.npy'), np.vstack(phis))
        # layer‑wise (optional)
        layer_deltas = tracex_layerwise(model, torch.from_numpy(X0[0]).float())
        torch.save(layer_deltas, os.path.join(args.output_dir,'layer_deltas.pt'))

    elif args.explainer=='shap':
        shap_vals = shap_explain(model, X0)
        np.save(os.path.join(args.output_dir,'shap.npy'), shap_vals)

    elif args.explainer=='lime':
        feature_names = list(pd.read_csv(args.data).drop('target',1).columns)
        lime_vals = lime_explain(model, X0, feature_names)
        np.save(os.path.join(args.output_dir,'lime.npy'), lime_vals)
