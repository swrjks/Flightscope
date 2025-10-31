import os
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------------------------
# Helper: load all sorties from a folder
# ---------------------------------
def load_sorties_from_folder(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    dfs = [pd.read_csv(f) for f in files]
    return dfs

# ---------------------------------
# Helper: sliding window generator
# ---------------------------------
def create_windows(df, features, label_col, window_size=50, step_size=10):
    X, y = [], []
    data = df[features].values
    labels = df[label_col].values
    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        X.append(data[start:end])
        # take majority label in the window
        window_labels = labels[start:end]
        uniq, counts = np.unique(window_labels, return_counts=True)
        y.append(uniq[np.argmax(counts)])
    return np.array(X), np.array(y)

# ---------------------------------
# Main preprocessing function
# ---------------------------------
def preprocess(train_dir, test_dir, out_dir, window_size=50, step_size=10):
    os.makedirs(out_dir, exist_ok=True)

    # feature columns (everything except time and label)
    feature_cols = [
        "altitude_ft", "vertical_speed_fpm", "ias_kt",
        "pitch_deg", "roll_deg", "heading_deg",
        "throttle_pct", "afterburner_on", "gear_down",
        "flaps_deg", "speedbrake_pct", "weight_on_wheels"
    ]
    label_col = "label"

    # load sorties
    train_dfs = load_sorties_from_folder(train_dir)
    test_dfs  = load_sorties_from_folder(test_dir)

    # encode labels
    le = LabelEncoder()
    le.fit([lab for df in train_dfs for lab in df[label_col].values])
    print("Classes:", le.classes_)

    # make windows
    X_train, y_train = [], []
    for df in train_dfs:
        X, y = create_windows(df, feature_cols, label_col, window_size, step_size)
        X_train.append(X); y_train.append(y)
    X_train = np.vstack(X_train); y_train = np.hstack(y_train)
    y_train = le.transform(y_train)

    X_test, y_test = [], []
    for df in test_dfs:
        X, y = create_windows(df, feature_cols, label_col, window_size, step_size)
        X_test.append(X); y_test.append(y)
    X_test = np.vstack(X_test); y_test = np.hstack(y_test)
    y_test = le.transform(y_test)

    # normalize (fit only on train)
    scaler = StandardScaler()
    n_samples, win_len, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_test_reshaped  = X_test.reshape(-1, n_features)

    scaler.fit(X_train_reshaped)
    X_train = scaler.transform(X_train_reshaped).reshape(n_samples, win_len, n_features)

    n_samples_test = X_test.shape[0]
    X_test = scaler.transform(X_test_reshaped).reshape(n_samples_test, win_len, n_features)

    # save
    np.savez_compressed(os.path.join(out_dir, "train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(out_dir, "test.npz"), X=X_test, y=y_test)
    print(f"Saved preprocessed data to {out_dir}/train.npz and {out_dir}/test.npz")

    # also save label encoder classes
    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        for c in le.classes_:
            f.write(c + "\n")

# ---------------------------------
# CLI
# ---------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, default="./dataset/fighter_regimes_synth/train")
    ap.add_argument("--test_dir",  type=str, default="./dataset/fighter_regimes_synth/test")
    ap.add_argument("--out_dir",   type=str, default="./dataset/processed")
    ap.add_argument("--window",    type=int, default=50, help="Sliding window size")
    ap.add_argument("--step",      type=int, default=10, help="Step size")
    args = ap.parse_args()

    preprocess(args.train_dir, args.test_dir, args.out_dir, args.window, args.step)
