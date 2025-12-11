#!/usr/bin/env python
"""Direct case lookup using simple os.path logic"""
import csv
import os

rows = []
with open('data/ProstateX-Findings-Test.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)[:5]

for row in rows:
    proxid = row.get('ProxID', '').strip()  # e.g., "ProstateX-0222"
    clin = row.get('ClinSig', '').strip()
    
    # Direct path construction
    case_dir = os.path.join('data', 'PROSTATEx', proxid)
    exists = os.path.isdir(case_dir)
    
    print(f"ProxID={proxid}, ClinSig={clin}, dir_exists={exists}")
    
    if exists:
        # List files in case dir
        for root, dirs, files in os.walk(case_dir):
            print(f"  {root}/")
            for f in files[:3]:
                print(f"    {f}")
