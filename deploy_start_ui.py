#!/usr/bin/env python3
"""
Production deployment test: Start Streamlit UI from deploy_clean/
"""
import subprocess
import sys
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent.parent / 'deploy_clean'

def main():
    print("=" * 80)
    print("PROSTATE ANALYZER - STREAMLIT WEB UI")
    print("=" * 80)
    
    # Verify deploy folder exists
    if not DEPLOY_DIR.exists():
        print(f"\n[ERROR] Deploy folder not found: {DEPLOY_DIR}")
        return 1
    
    print(f"\n[1] Deploy folder: {DEPLOY_DIR}")
    print(f"\n[2] Starting Streamlit UI...")
    print(f"    Command: python -m streamlit run webapp/streamlit_app.py")
    print(f"\n    UI will open at: http://localhost:8501")
    print(f"\n    Press Ctrl+C to stop")
    print(f"\n" + "=" * 80 + "\n")
    
    try:
        import os
        os.chdir(DEPLOY_DIR)
        
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            'webapp/streamlit_app.py'
        ])
    except KeyboardInterrupt:
        print("\n\n[OK] UI stopped.")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Failed to start UI: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
