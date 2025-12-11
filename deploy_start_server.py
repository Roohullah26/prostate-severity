#!/usr/bin/env python3
"""
Production deployment test: Start FastAPI inference server from deploy_clean/
"""
import subprocess
import sys
import time
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parent.parent / 'deploy_clean'

def main():
    print("=" * 80)
    print("PROSTATE ANALYZER - PRODUCTION DEPLOYMENT")
    print("=" * 80)
    
    # Verify deploy folder exists
    if not DEPLOY_DIR.exists():
        print(f"\n[ERROR] Deploy folder not found: {DEPLOY_DIR}")
        return 1
    
    print(f"\n[1] Deploy folder: {DEPLOY_DIR}")
    print(f"    Contents:")
    for item in sorted(DEPLOY_DIR.iterdir()):
        if item.is_dir():
            print(f"      ├─ {item.name}/")
        else:
            size = item.stat().st_size
            if size > 1024*1024:
                print(f"      ├─ {item.name} ({size/(1024*1024):.1f} MB)")
            else:
                print(f"      ├─ {item.name}")
    
    print(f"\n[2] Starting FastAPI inference server...")
    print(f"    Command: python -m uvicorn webapp.fastapi_server:app --host 0.0.0.0 --port 8000")
    print(f"\n    Server will be available at:")
    print(f"      • REST API: http://localhost:8000")
    print(f"      • API Docs: http://localhost:8000/docs")
    print(f"      • ReDoc: http://localhost:8000/redoc")
    print(f"\n    Press Ctrl+C to stop the server")
    print(f"\n" + "=" * 80 + "\n")
    
    try:
        # Change to deploy_clean directory and start server
        import os
        os.chdir(DEPLOY_DIR)
        
        # Start uvicorn server
        subprocess.run([
            sys.executable, '-m', 'uvicorn',
            'webapp.fastapi_server:app',
            '--host', '0.0.0.0',
            '--port', '8000',
            '--reload'
        ])
    except KeyboardInterrupt:
        print("\n\n[OK] Server stopped.")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Failed to start server: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
