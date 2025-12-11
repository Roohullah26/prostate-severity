@echo off
cd /d "D:\prostate project\prostate-severity"
call venv\Scripts\activate.bat
python scripts/eval_clinsig_confusion.py --csv data/ProstateX-Findings-Test.csv --model models/baseline_real_t2_adc_3s_ep1.pth --limit 100 --out results
pause
