import pandas as pd
from pathlib import Path

# Read CSVs from the repository data/ folder when available
repo_root = Path(__file__).resolve().parents[1]
data_dir = repo_root / "data"

meta_fp = data_dir / "metadata.csv"
findings_fp = data_dir / "ProstateX-Findings-Test.csv"

if not meta_fp.exists() or not findings_fp.exists():
	# fallback to current folder
	meta_fp = repo_root / "metadata.csv"
	findings_fp = repo_root / "ProstateX-Findings-Test.csv"

meta = pd.read_csv(meta_fp)
findings = pd.read_csv(findings_fp)

meta["Subject ID"] = meta["Subject ID"].str.replace("PROSTATEX", "ProstateX", case=False)
findings["ProxID"] = findings["ProxID"].str.replace("PROSTATEX", "ProstateX", case=False)

merged = pd.merge(findings, meta, left_on="ProxID", right_on="Subject ID", how="left")
merged = merged.drop_duplicates(subset=["ProxID", "fid", "Series UID"])

out_fp = repo_root / "merged_data.csv"
merged.to_csv(out_fp, index=False)
print(f"saved {out_fp}")
