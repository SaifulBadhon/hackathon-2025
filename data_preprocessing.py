from pathlib import Path
import re
import pandas as pd

# === CONFIG ===
ROOT = Path("E:/Hackathon UNT 2025/pads-parkinsons-disease-smartwatch-dataset-1.0.0/pads-parkinsons-disease-smartwatch-dataset-1.0.0/movement/timeseries/")  # change to your folder
print(ROOT )
N_COLS = 7
SPLIT_EVERY_1024 = False

# Regex to pull subject/activity from filename (handles several patterns)
# Examples it handles:
#   001.txt  -> subject=001
#   001_crossArms_Leftwrist.txt -> subject=001, activity=crossArms_Leftwrist
#   crossArms_Leftwrist_001.txt -> subject=001, activity=crossArms_Leftwrist
SUBJECT_PAT = re.compile(r"(?<!\d)(\d{3})(?!\d)")
ACTIVITY_PATNERS = [
    re.compile(r"^\s*(\D.+?)_(\d{3})\s*$"),          # activity_001
    re.compile(r"^\s*(\d{3})_(\D.+?)\s*$"),          # 001_activity
    re.compile(r"^\s*(\D.+?)\s*$"),                  # activity only
]

def parse_name(stem: str):
    """Return (subject_id, activity or None) from filename stem."""
    subject = None
    m = SUBJECT_PAT.search(stem)
    if m: subject = m.group(1)

    activity = None
    # remove subject digits to try to isolate activity
    cleaned = SUBJECT_PAT.sub("", stem).strip("_- ")
    for pat in ACTIVITY_PATNERS:
        mm = pat.match(cleaned)
        if mm:
            # choose text group that is not the subject digits
            for g in mm.groups():
                if g and not g.isdigit():
                    activity = g.strip("_- ")
                    break
            break
    return subject, activity

def to_cols(n):
    return [f"v{i}" for i in range(1, n+1)]

all_rows = []

txt_files = sorted(ROOT.rglob("*.txt"))
print(f"Found {len(txt_files)} .txt files under {ROOT.resolve()}")
for f in txt_files:
    print(f)
    subject, activity = parse_name(f.stem)
    print(subject, activity)
    # read as CSV with no header
    df = pd.read_csv(f, header=None, sep=r",\s*", engine="python")
    # sanity: pad/trim columns to N_COLS if needed
    if df.shape[1] != N_COLS:
        # try to coerce by splitting on comma strictly
        df = pd.read_csv(f, header=None, sep=",", engine="python")
    if df.shape[1] != N_COLS:
        raise ValueError(f"{f.name}: expected {N_COLS} columns, got {df.shape[1]}")

    df.columns = to_cols(N_COLS)
    df.insert(0, "subject_id", subject if subject else "")
    df.insert(1, "activity", activity if activity else "")

    out_csv = f.with_suffix(".csv")
    df.to_csv(out_csv, index=False)

    all_rows.append(df)

if all_rows:
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(ROOT / "all_subjects.csv", index=False)
    print(f"Wrote combined CSV: {ROOT / 'all_subjects.csv'}")
else:
    print("No rows aggregated; check your directory or file patterns.")
