import pandas as pd
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split

def downsample_if_needed(arr, target_len=1024):
    """
    If array length is 2048, average every 2 points to make 1024.
    Otherwise, return unchanged.
    """
    if arr.shape[0] == 2048:
        # reshape to (1024, 2, n_features) and mean along axis 1
        arr = arr.reshape(target_len, 2, arr.shape[1]).mean(axis=1)
    return arr
def create_samples_labels(df):


    feature_cols = ["v2","v3","v4","v5","v6","v7"]
    print(np.unique(df['wrist']))
    left_df = df[df['wrist']=='LeftWrist']
    print(left_df)
    right_df = df[df['wrist']=='RightWrist']
    activities = np.unique(df['activity_clean'])
    subject_ids = np.unique(df['id'])
    labels = []
    X_left_list  = []
    X_right_list = []
    y_list       = []
    sample_subject_ids = []
    for sub in subject_ids:
        # Filter subject data
        temp_left  = left_df[left_df['id'] == sub]
        temp_right = right_df[right_df['id'] == sub]

        print(f"\n=== Subject {sub} ===")
        print(f"Left wrist samples: {len(temp_left)}, Right wrist samples: {len(temp_right)}")

        for activity in activities:
            # Left and right activity data for this subject
            X_left  = temp_left[temp_left['activity_clean'] == activity]
            X_right = temp_right[temp_right['activity_clean'] == activity]

            X_left_np  = X_left[feature_cols].to_numpy(dtype=np.float32)
            X_right_np = X_right[feature_cols].to_numpy(dtype=np.float32)

            X_left_np  = downsample_if_needed(X_left_np)
            X_right_np = downsample_if_needed(X_right_np)
            if X_left_np.shape[0] != X_right_np.shape[0]:
                continue
            X_left_list.append(X_left_np)
            X_right_list.append(X_right_np)
            y_list.append(activity)
            sample_subject_ids.append(sub)
    return X_left_list, X_right_list, y_list,sample_subject_ids


# =========
# CONFIG
# =========
SEQ_LEN      = 1024        # time steps
PATCH_LEN    = 8           # optional: reduce T via patching (T' = T / PATCH_LEN)
EMBED_DIM    = 128
FF_DIM       = 256
NUM_HEADS    = 4
NUM_LAYERS   = 4
DROPOUT      = 0.1
BATCH_SIZE   = 16
EPOCHS       = 25
LR           = 1e-3
RANDOM_SEED  = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========
# INPUTS (you already have these from your loop)
#   X_left_list:  List[np.ndarray (T,F)]
#   X_right_list: List[np.ndarray (T,F)]
#   y_list:       List[str]
# Replace the following placeholders with your real lists
# =========
# X_left_list  = ...
# X_right_list = ...
# y_list       = ...

# =========
# HELPERS
# =========
def encode_labels(y_list):
    classes = sorted(set(y_list))
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    y_idx = np.array([label2id[a] for a in y_list], dtype=np.int64)
    return y_idx, label2id, id2label

def compute_norm_stats(XA, XB):
    A = np.concatenate([x for x in XA], axis=0)  # (sum_T, F)
    B = np.concatenate([x for x in XB], axis=0)
    ALL = np.vstack([A, B])
    mean = ALL.mean(axis=0).astype(np.float32)
    std  = ALL.std(axis=0).astype(np.float32)
    std[std == 0.0] = 1.0
    return mean, std

def subjectwise_split(ids, ratio=0.2, seed=RANDOM_SEED):
    uniq = np.array(sorted(set(ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * ratio))
    val = set(uniq[:n_val])
    train = set(uniq[n_val:])
    return train, val

# =========
# DATASET
# =========
class TwoStreamDataset(Dataset):
    def __init__(self, XL, XR, y, mean, std, patch_len=1):
        self.XL = XL
        self.XR = XR
        self.y  = y
        self.m  = torch.tensor(mean, dtype=torch.float32)
        self.s  = torch.tensor(std,  dtype=torch.float32)
        self.patch_len = patch_len

    def _norm(self, x):
        # x: (T,F) np -> torch
        x = torch.tensor(x, dtype=torch.float32)
        return (x - self.m) / self.s

    @staticmethod
    def _patch(x, patch_len):
        # x: (T,F) -> (T', patch_len*F) by non-overlapping windows
        if patch_len == 1:
            return x
        T, F = x.shape
        assert T % patch_len == 0, "T must be divisible by PATCH_LEN"
        Tnew = T // patch_len
        x = x.view(Tnew, patch_len, F).mean(dim=1)  # avg pool over patch_len (simple)
        return x

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        xl = self._norm(self.XL[idx])
        xr = self._norm(self.XR[idx])
        xl = self._patch(xl, self.patch_len)  # (T', F)
        xr = self._patch(xr, self.patch_len)
        y  = torch.tensor(self.y[idx], dtype=torch.long)
        return xl, xr, y

# =========
# MODEL
# =========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class StreamEncoder(nn.Module):
    """
    Projects (features) -> (embed), adds positional enc, runs TransformerEncoder,
    and returns a pooled representation via a learnable [CLS] token.
    """
    def __init__(self, in_dim, embed_dim, ff_dim, n_heads, n_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos = PositionalEncoding(embed_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim))  # (1,1,D)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, T, F)
        B = x.size(0)
        x = self.proj(x)             # (B,T,D)
        x = self.pos(x)              # add PE
        cls = self.cls.expand(B, -1, -1)  # (B,1,D)
        x = torch.cat([cls, x], dim=1)    # prepend CLS → (B,1+T,D)
        h = self.encoder(x)          # (B,1+T,D)
        cls_out = self.ln(h[:, 0, :])# (B,D)
        return cls_out

class TwoStreamTransformer(nn.Module):
    def __init__(self, in_dim, num_classes, embed_dim=EMBED_DIM, ff_dim=FF_DIM,
                 n_heads=NUM_HEADS, n_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.left_enc  = StreamEncoder(in_dim, embed_dim, ff_dim, n_heads, n_layers, dropout)
        self.right_enc = StreamEncoder(in_dim, embed_dim, ff_dim, n_heads, n_layers, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, num_classes)
        )

    def forward(self, xl, xr):
        # xl/xr: (B, T', F)
        zl = self.left_enc(xl)
        zr = self.right_enc(xr)
        z = torch.cat([zl, zr], dim=1)    # (B, 2D)
        logits = self.head(z)             # (B, C)
        return logits

# =========
# TRAINING LOOP
# =========
def run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    if train: model.train()
    else:     model.eval()

    total, correct, loss_sum = 0, 0, 0.0
    for xl, xr, yb in loader:
        xl = xl.to(device, non_blocking=True)
        xr = xr.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if train: opt.zero_grad()
        logits = model(xl, xr)
        loss = crit(logits, yb)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        loss_sum += loss.item() * yb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return loss_sum / total, correct / total

def get_combined_df(df,size=70):
        # Filter Healthy and Parkinson's rows
    healthy_df = df[df['condition'] == "Healthy"]
    parkinsons_df = df[df['condition'] == "Parkinson's"]

    # Get unique Parkinson's subject IDs
    parkinsons_subjects = parkinsons_df['id'].unique()

    # Randomly select 70 unique Parkinson's subject IDs
    sampled_ids = np.random.choice(parkinsons_subjects, size=size, replace=False)

    # Get all rows for those selected Parkinson's subjects
    sampled_parkinsons_df = parkinsons_df[parkinsons_df['id'].isin(sampled_ids)]
    combined_df = pd.concat([healthy_df, sampled_parkinsons_df], ignore_index=True)

    return combined_df
main_path = 'E:/Hackathon UNT 2025/pads-parkinsons-disease-smartwatch-dataset-1.0.0/pads-parkinsons-disease-smartwatch-dataset-1.0.0/'
path = 'filtered_all_subjects_with_features.csv'
df =  pd.read_csv(main_path+path, index_col=0)
df = df[df['condition'].isin(["Healthy", "Parkinson's"])]
df = get_combined_df(df,size=70)
print(df.columns)
X_left_list, X_right_list, y_list, sample_subject_ids = create_samples_labels(df)
print(f"Total paired samples: {len(y_list)}")
print("Example left/right shape:", X_left_list[0].shape, X_right_list[0].shape)
print("Example label:", y_list)
# Dims
T, F = X_left_list[0].shape

# Encode labels + subject-wise split (optional)
y_idx, label2id, id2label = encode_labels(y_list)
# y_idx, label2id, id2label = subjectwise_split(y_list)
print("Classes:", label2id)

idx = np.arange(len(y_idx))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_SEED, stratify=y_idx)

XL_tr = [X_left_list[i] for i in tr_idx]
XR_tr = [X_right_list[i] for i in tr_idx]
y_tr  = y_idx[tr_idx]
XL_va = [X_left_list[i] for i in va_idx]
XR_va = [X_right_list[i] for i in va_idx]
val_subject_ids = [sample_subject_ids[i] for i in va_idx]
print(len(val_subject_ids ), len(XL_va))
y_va  = y_idx[va_idx]

    # Normalization from TRAIN only (over both streams)
mean, std = compute_norm_stats(XL_tr, XR_tr)
print('mean',mean)
print('std',std)
# Patch length (reduces T for memory/speed). With SEQ_LEN=1024 and PATCH_LEN=8 → T'=128.
patch_len = PATCH_LEN
assert T % patch_len == 0, "Sequence length must be divisible by PATCH_LEN."

# Datasets/Loaders
train_ds = TwoStreamDataset(XL_tr, XR_tr, y_tr, mean, std, patch_len=patch_len)
val_ds   = TwoStreamDataset(XL_va, XR_va, y_va, mean, std, patch_len=patch_len)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# Model / Loss / Optim
model = TwoStreamTransformer(in_dim=F, num_classes=len(label2id)).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

best = 0.0
for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, None)
        print(f"Epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")
        if va_acc > best:
            best = va_acc
            #
            print(f"Saved best (val acc {best:.4f})")
# torch.save(model.state_dict(), "two_stream_transformer_hc_pd.pt")
    # quick sanity predictions
model.eval()
all_preds = []
all_true = []
all_val_ids = []

val_subject_ids = list(val_subject_ids)  # make it indexable

with torch.no_grad():
    i = 0
    for xl, xr, yb in val_loader:
        xl = xl.to(device)
        xr = xr.to(device)
        yb = yb.to(device)

        logits = model(xl, xr)
        preds = logits.argmax(1)

        batch_size = yb.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(yb.cpu().numpy())
        all_val_ids.extend(val_subject_ids[i:i+batch_size])  # FIXED LINE
        i += batch_size


# Convert to numpy arrays
all_preds = np.array(all_preds)
all_true  = np.array(all_true)

# Optional: map numeric → string labels
pred_labels = [id2label[i] for i in all_preds]
true_labels = [id2label[i] for i in all_true]

val_results = pd.DataFrame({
    "true": [id2label[i] for i in all_true],
    "pred": [id2label[i] for i in all_preds],
    "subject_id": all_val_ids
})
val_results.to_csv("validation_results.csv", index=False)
# Get condition per subject from df
id2cond = df.groupby("id")["condition"].first().to_dict()

val_results["condition"] = val_results["subject_id"].map(id2cond)
healthy_df = val_results[val_results["condition"] == "Healthy"]
parkinson_df = val_results[val_results["condition"] == "Parkinson's"]

from sklearn.metrics import classification_report, confusion_matrix

print("\n✅ Healthy Subject Evaluation")
print(classification_report(healthy_df["true"], healthy_df["pred"]))
print(confusion_matrix(healthy_df["true"], healthy_df["pred"]))

print("\n✅ Parkinson's Subject Evaluation")
print(classification_report(parkinson_df["true"], parkinson_df["pred"]))
print(confusion_matrix(parkinson_df["true"], parkinson_df["pred"]))



# If not already defined
# all_true, all_preds = np.array(...)

# Compute overall accuracy
acc = accuracy_score(all_true, all_preds)
print(f"\nOverall Validation Accuracy: {acc*100:.2f}%")

# Per-class precision, recall, F1
prec, rec, f1, support = precision_recall_fscore_support(all_true, all_preds, average=None)

# Build DataFrame for per-class metrics
class_results = pd.DataFrame({
    "Class": [id2label[i] for i in range(len(prec))],
    "Precision": prec,
    "Recall": rec,
    "F1-Score": f1,
    "Support": support
})
print("\nPer-class metrics:")
print(class_results)

# Macro / weighted averages
report = classification_report(all_true, all_preds, target_names=[id2label[i] for i in range(len(id2label))])
print("\nDetailed classification report:\n", report)

# Confusion matrix
cm = confusion_matrix(all_true, all_preds)
print("Confusion Matrix:\n", cm)

# Optionally save to CSV
class_results.to_csv("val_classification_results_hc_pd.csv", index=False)
