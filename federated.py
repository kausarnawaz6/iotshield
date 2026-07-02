import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from copy import deepcopy

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
DATA_DIR    = r"E:\iotj_revision\data"
NUM_ROUNDS  = 5       # federated communication rounds
LOCAL_EPOCH = 3       # local training epochs per round
LEARNING_RATE = 0.01
SAVE_PATH   = r"E:\iotj_revision\roc_federated.png"

# ── 1. LOAD PER-DEVICE DATA (each device = one federated client) ───────────────
print("Loading per-device data...")
device_data = {}

for device in sorted(os.listdir(DATA_DIR)):
    device_path = os.path.join(DATA_DIR, device)
    if not os.path.isdir(device_path):
        continue
    csv_files = glob.glob(os.path.join(device_path, "**", "*.csv"), recursive=True)
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            parent = os.path.basename(os.path.dirname(f)).lower()
            fname  = os.path.basename(f).lower()
            is_benign = ("benign" in parent) or ("benign" in fname)
            df["label"] = 0 if is_benign else 1
            dfs.append(df)
        except Exception as e:
            print(f"  Skip {f}: {e}")
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        X = combined.drop(columns=["label"]).select_dtypes(include=[np.number]).fillna(0)
        y = combined["label"]
        device_data[device] = (X.values, y.values)
        print(f"  {device[:35]}: {len(y)} rows  "
              f"(benign={int((y==0).sum())}, attack={int((y==1).sum())})")

print(f"\nTotal clients (devices): {len(device_data)}\n")

# ── 2. SIMPLE NEURAL NET IN NUMPY (so we can do FedAvg on weights) ────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0, x)

def init_weights(input_dim, hidden=64):
    """Xavier initialisation"""
    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden) * np.sqrt(2.0/input_dim)
    b1 = np.zeros(hidden)
    W2 = np.random.randn(hidden, 32) * np.sqrt(2.0/hidden)
    b2 = np.zeros(32)
    W3 = np.random.randn(32, 1) * np.sqrt(2.0/32)
    b3 = np.zeros(1)
    return {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3}

def forward(X, weights):
    z1 = X @ weights["W1"] + weights["b1"]
    a1 = relu(z1)
    z2 = a1 @ weights["W2"] + weights["b2"]
    a2 = relu(z2)
    z3 = a2 @ weights["W3"] + weights["b3"]
    out = sigmoid(z3).flatten()
    return out, (X, a1, a2)

def compute_loss(y_true, y_pred):
    eps = 1e-9
    return -np.mean(y_true * np.log(y_pred+eps) + (1-y_true)*np.log(1-y_pred+eps))

def local_train(weights, X, y, epochs=3, lr=0.01, batch=512):
    """Train locally for a few epochs using mini-batch SGD"""
    w = deepcopy(weights)
    n = len(X)
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch):
            end   = min(start+batch, n)
            Xb    = X[idx[start:end]]
            yb    = y[idx[start:end]].astype(float)
            preds, (inp, a1, a2) = forward(Xb, w)

            # Backprop (chain rule)
            dL   = (preds - yb) / len(yb)
            dW3  = (a2.T @ dL.reshape(-1,1))
            db3  = dL.sum()
            da2  = (dL.reshape(-1,1) @ w["W3"].T) * (a2 > 0)
            dW2  = (a1.T @ da2)
            db2  = da2.sum(axis=0)
            da1  = (da2 @ w["W2"].T) * (a1 > 0)
            dW1  = (inp.T @ da1)
            db1  = da1.sum(axis=0)

            # SGD update
            for key, grad in [("W1",dW1),("b1",db1),("W2",dW2),
                               ("b2",db2),("W3",dW3),("b3",db3)]:
                w[key] -= lr * grad
    return w

def fed_avg(local_weights_list, n_samples_list):
    """Weighted average of client weights by number of samples"""
    total = sum(n_samples_list)
    avg_weights = {}
    for key in local_weights_list[0]:
        avg_weights[key] = sum(
            w[key] * (n / total)
            for w, n in zip(local_weights_list, n_samples_list)
        )
    return avg_weights

def predict(weights, X, threshold=0.5):
    probs, _ = forward(X, weights)
    return probs, (probs >= threshold).astype(int)

# ── 3. PREPARE SCALED TRAIN/TEST SPLITS PER DEVICE ────────────────────────────
print("Preparing per-device train/test splits...")
scaler = StandardScaler()

# Fit scaler on all data combined
all_X = np.vstack([v[0] for v in device_data.values()])
scaler.fit(all_X)

client_train = {}
client_test  = {}
X_test_all, y_test_all = [], []

for device, (X, y) in device_data.items():
    Xs = scaler.transform(X)
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=0.30, stratify=y, random_state=42)
    client_train[device] = (Xtr, ytr)
    client_test[device]  = (Xte, yte)
    X_test_all.append(Xte)
    y_test_all.append(yte)

X_test_global = np.vstack(X_test_all)
y_test_global = np.concatenate(y_test_all)
input_dim = X_test_global.shape[1]
print(f"Global test set: {len(y_test_global)} samples, {input_dim} features\n")

# ── 4. FEDERATED LEARNING LOOP ────────────────────────────────────────────────
print(f"Starting Federated Learning: {NUM_ROUNDS} rounds, "
      f"{len(client_train)} clients, {LOCAL_EPOCH} local epochs each\n")

global_weights = init_weights(input_dim)
round_aucs = []

for rnd in range(1, NUM_ROUNDS + 1):
    local_weights_list = []
    n_samples_list     = []

    # Each client trains locally
    for device, (Xtr, ytr) in client_train.items():
        local_w = local_train(
            global_weights, Xtr, ytr,
            epochs=LOCAL_EPOCH, lr=LEARNING_RATE)
        local_weights_list.append(local_w)
        n_samples_list.append(len(ytr))

    # Server aggregates via FedAvg
    global_weights = fed_avg(local_weights_list, n_samples_list)

    # Evaluate global model on held-out test data
    probs, preds = predict(global_weights, X_test_global)
    auc = roc_auc_score(y_test_global, probs)
    round_aucs.append(auc)
    tn, fp, fn, tp = confusion_matrix(y_test_global, preds).ravel()
    fnr = fn / (fn + tp) if (fn+tp) > 0 else 0
    print(f"  Round {rnd}/{NUM_ROUNDS} | AUC={auc:.4f} | FNR={fnr*100:.2f}%"
          f" | Clients={len(client_train)}")

# ── 5. FINAL EVALUATION ───────────────────────────────────────────────────────
probs_final, preds_final = predict(global_weights, X_test_global)
auc_final = roc_auc_score(y_test_global, probs_final)
tn, fp, fn, tp = confusion_matrix(y_test_global, preds_final).ravel()
fnr_final = fn / (fn + tp) if (fn+tp) > 0 else 0
fpr_final = fp / (fp + tn) if (fp+tn) > 0 else 0

print(f"\n{'='*50}")
print(f"  FEDERATED LEARNING — FINAL RESULTS")
print(f"{'='*50}")
print(f"  Rounds          : {NUM_ROUNDS}")
print(f"  Clients         : {len(client_train)} IoT devices")
print(f"  AUC             : {auc_final:.4f}")
print(f"  FNR             : {fnr_final:.4f}  ({fnr_final*100:.2f}%)")
print(f"  FPR             : {fpr_final:.4f}  ({fpr_final*100:.2f}%)")
print(f"{'='*50}")
print(classification_report(y_test_global, preds_final,
                             target_names=["benign","attack"]))

# ── 6. PLOTS ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# ROC curve
fpr_arr, tpr_arr, _ = roc_curve(y_test_global, probs_final)
ax1.plot(fpr_arr, tpr_arr, color="darkorange", lw=2,
         label=f"Federated Model (AUC={auc_final:.3f})")
ax1.plot([0,1],[0,1],"k--",lw=1)
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC — Phase 4 Federated Learning (N-BaIoT)")
ax1.legend(loc="lower right")

# Convergence curve
ax2.plot(range(1, NUM_ROUNDS+1), round_aucs,
         marker="o", color="steelblue", lw=2)
ax2.set_xlabel("Communication Round")
ax2.set_ylabel("AUC")
ax2.set_title("Federated Convergence Across Rounds")
ax2.set_ylim(0, 1.05)
ax2.set_xticks(range(1, NUM_ROUNDS+1))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150)
plt.show()
print(f"\nPlots saved to {SAVE_PATH}")