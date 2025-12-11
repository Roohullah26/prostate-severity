#!/usr/bin/env python
"""Quick debug to see what happens with first few rows"""
import csv
import glob
import os

# Read CSV
rows = []
with open('data/ProstateX-Findings-Test.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)[:10]

print(f"Processing {len(rows)} rows\n")

for i, row in enumerate(rows):
    proxid = row.get('ProxID', '').strip()
    clin = row.get('ClinSig', '').strip()
    print(f"Row {i}: ProxID={proxid}, ClinSig={clin}")
    
    # Try to find case
    pattern = f"data/PROSTATEx/**/{proxid}*"
    matches = glob.glob(pattern, recursive=True)
    print(f"  Glob matches: {len(matches)}")
    
    case_dir = None
    for m in matches:
        if os.path.isdir(m):
            case_dir = m
            break
    
    if case_dir:
        print(f"  Found case dir: {case_dir}")
        # List contents
        for root, dirs, files in os.walk(case_dir):
            for f in files[:5]:
                print(f"    {os.path.join(root, f)}")
    else:
        print(f"  No case dir found")
    print()
