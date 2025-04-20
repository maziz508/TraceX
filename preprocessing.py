import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

def load_and_preprocess(path, test_size=0.2, val_size=0.1, random_state=42):
    """
    Reads CSV, splits into train/val/test, applies one-hot + z‑scale, and saves scalers/encoders.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    df = pd.read_csv(path)
    y = df['target'].values
    X_num = df.select_dtypes(include=['int64','float64']).drop('target', axis=1)
    X_cat = df.select_dtypes(include=['object','category'])

    # split off test
    Xn, Xn_test, Xc, Xc_test, y_trainval, y_test = train_test_split(
        X_num, X_cat, y, test_size=test_size, random_state=random_state
    )
    # split train/val
    val_frac = val_size / (1 - test_size)
    Xn_train, Xn_val, Xc_train, Xc_val, y_train, y_val = train_test_split(
        Xn, Xc, y_trainval, test_size=val_frac, random_state=random_state
    )

    # fit numeric scaler
    scaler = StandardScaler().fit(Xn_train)
    Xn_train_s = scaler.transform(Xn_train)
    Xn_val_s   = scaler.transform(Xn_val)
    Xn_test_s  = scaler.transform(Xn_test)
    joblib.dump(scaler, 'outputs/scaler.joblib')

    # fit categorical encoder
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(Xc_train)
    Xc_train_o = encoder.transform(Xc_train)
    Xc_val_o   = encoder.transform(Xc_val)
    Xc_test_o  = encoder.transform(Xc_test)
    joblib.dump(encoder, 'outputs/encoder.joblib')

    # combine
    X_train = np.hstack([Xn_train_s, Xc_train_o])
    X_val   = np.hstack([Xn_val_s,   Xc_val_o])
    X_test  = np.hstack([Xn_test_s,  Xc_test_o])

    return X_train, X_val, X_test, y_train, y_val, y_test
