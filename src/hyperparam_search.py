import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from preprocessing import load_and_preprocess
from models import MLP, CNN, TransformerModel, AutoencoderClassifier

def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float())
        preds = (torch.sigmoid(logits) > 0.5).numpy()
    return accuracy_score(y, preds)

def search(model_name, param_grid, data_path):
    X_train, X_val, _, y_train, y_val, _ = load_and_preprocess(data_path)
    input_dim = X_train.shape[1]
    best = {'params':None, 'acc':0}
    for params in ParameterGrid(param_grid):
        # build model with these params
        if model_name=='mlp':
            model = MLP(input_dim, hidden=params['hidden_units'])
        elif model_name=='cnn':
            model = CNN(input_dim, channels=params['channels'])
        elif model_name=='transformer':
            model = TransformerModel(
                input_dim,
                d_model=params['d_model'],
                heads=params['heads'],
                layers=params['layers']
            )
        else:
            model = AutoencoderClassifier(input_dim, bottleneck=params['bottleneck'])

        opt = optim.Adam(model.parameters(), lr=params['lr'])
        loss_fn = nn.BCEWithLogitsLoss()

        # one epoch
        model.train()
        logits = model(torch.from_numpy(X_train).float())
        loss = loss_fn(logits, torch.from_numpy(y_train).float())
        opt.zero_grad(); loss.backward(); opt.step()

        # evaluate
        acc = evaluate(model, X_val, y_val)
        if acc>best['acc']:
            best = {'acc':acc, 'params':params}
    return best

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp','cnn','transformer','autoencoder'], required=True)
    parser.add_argument('--data', default='data/loan_data.csv')
    args = parser.parse_args()

    grid = {
      'mlp': {
        'hidden_units': [64,128],
        'lr': [1e-3,5e-4]
      },
      'cnn': {
        'channels': [16,32],
        'lr': [1e-3,5e-4]
      },
      'transformer': {
        'd_model': [64,128],
        'heads': [2,4],
        'layers': [2,3],
        'lr': [1e-3,5e-4]
      },
      'autoencoder': {
        'bottleneck': [16,32],
        'lr': [1e-3,5e-4]
      }
    }

    best = search(args.model, grid[args.model], args.data)
    print(f"Best for {args.model}: {best}")
