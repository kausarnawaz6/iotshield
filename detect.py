import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
data_dir = r"E:\iotj_revision\data"
device_folders = [f for f in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, f))]

print(f"Found {len(device_folders)} device folders")

dfs = []
for device in device_folders:
    device_path = os.path.join(data_dir, device)
    csv_files = glob.glob(os.path.join(device_path, "**", "*.csv"), recursive=True)
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df["device"] = device
            # Check parent subfolder name for label
            parent = os.path.basename(os.path.dirname(f)).lower()
            fname  = os.path.basename(f).lower()
            is_benign = ("benign" in parent) or ("benign" in fname)
            df["label"] = 0 if is_benign else 1
            dfs.append(df)
            print(f"  {'BENIGN' if is_benign else 'ATTACK'} | {device[:20]} | {os.path.basename(f)} | {len(df)} rows")
        except Exception as e:
            print(f"  SKIP {f}: {e}")

full_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows    : {full_df.shape[0]}")
print(f"Label counts  :")
print(full_df["label"].value_counts().rename({0:'benign',1:'attack'}))

# ── 2. PREPARE FEATURES ───────────────────────────────────────────────────────
X = full_df.drop(columns=["label","device"])
y = full_df["label"]
X = X.select_dtypes(include=[np.number]).fillna(0)

print(f"\nFeature count : {X.shape[1]}")

# ── 3. TRAIN/TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42)
print(f"Train samples : {len(X_train)}")
print(f"Test  samples : {len(X_test)}")

# ── 4. TRAIN DETECTOR ────────────────────────────────────────────────────────
print("\nTraining detector... (may take 2-3 minutes)")
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

# ── 5. EVALUATE ──────────────────────────────────────────────────────────────
y_scores = clf.predict_proba(X_test)[:,1]
y_pred   = clf.predict(X_test)

auc = roc_auc_score(y_test, y_scores)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fnr = fn / (fn + tp)
fpr = fp / (fp + tn)

print(f"\n{'='*45}")
print(f"  AUC  : {auc:.4f}")
print(f"  FNR  : {fnr:.4f}  ({fnr*100:.2f}%)")
print(f"  FPR  : {fpr:.4f}  ({fpr*100:.2f}%)")
print(f"{'='*45}")
print(classification_report(y_test, y_pred, target_names=["benign","attack"]))

# ── 6. ROC CURVE ─────────────────────────────────────────────────────────────
fpr_arr, tpr_arr, _ = roc_curve(y_test, y_scores)
plt.figure(figsize=(7,5))
plt.plot(fpr_arr, tpr_arr, color="steelblue", lw=2,
         label=f"N-BaIoT Validation (AUC = {auc:.3f})")
plt.plot([0,1],[0,1],"k--",lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Phase 3 Detection on N-BaIoT")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(r"E:\iotj_revision\roc_nbaiot.png", dpi=150)
plt.show()
print("\nDone! ROC saved to E:\\iotj_revision\\roc_nbaiot.png")