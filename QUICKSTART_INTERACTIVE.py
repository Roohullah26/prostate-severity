"""
QUICK START GUIDE - Tumor Size Prediction & Severity Analysis

This script demonstrates the complete workflow for tumor prediction.
Run this to see the system in action!
"""

import os
import sys
import subprocess
from pathlib import Path


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def run_command(cmd, description, shell=True):
    """Run command and print output"""
    print(f"▶️  {description}")
    print(f"   Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False


def main():
    """Run quickstart"""
    project_root = Path(__file__).parent.parent
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  🏥 TUMOR SIZE PREDICTION SYSTEM - QUICK START".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print("""
This quick start demonstrates:
  1. Running comprehensive tests
  2. Executing the full pipeline demo
  3. Testing the API endpoints
  4. Generating visualizations

Choose an option:
""")
    
    print("Available Commands:")
    print("  1. Run comprehensive tests")
    print("  2. Run full pipeline demo")
    print("  3. Test API endpoints")
    print("  4. View system status")
    print("  5. Run all")
    print("  0. Exit")
    
    choice = input("\nEnter your choice (0-5): ").strip()
    
    if choice == '1':
        print_section("Running Comprehensive Tests")
        os.chdir(project_root)
        os.system(f"{sys.executable} scripts/run_comprehensive_test.py")
    
    elif choice == '2':
        print_section("Running Full Pipeline Demo")
        os.chdir(project_root)
        os.system(f"{sys.executable} scripts/demo_full_pipeline.py")
    
    elif choice == '3':
        print_section("Testing API Endpoints")
        print("Starting FastAPI server...")
        print("In another terminal, run:")
        print(f"  python scripts/test_api_client.py --test")
        print("\nOr to start server with model:")
        print(f"  python webapp/fastapi_server.py --host 0.0.0.0 --port 8000")
    
    elif choice == '4':
        print_section("System Status")
        print("Python Version:", sys.version)
        print("Project Root:", project_root)
        
        # Check key files
        print("\nKey Components:")
        components = [
            ('src/size_predictor_model.py', 'Size Predictor Model'),
            ('src/bbox_utils.py', 'Bounding Box Utils'),
            ('src/visualization_enhanced.py', 'Visualization'),
            ('webapp/fastapi_server.py', 'FastAPI Server'),
            ('scripts/demo_full_pipeline.py', 'Demo Script'),
        ]
        
        for filepath, name in components:
            full_path = project_root / filepath
            if full_path.exists():
                print(f"  ✅ {name}")
            else:
                print(f"  ❌ {name} - NOT FOUND")
    
    elif choice == '5':
        print_section("Running All Tests")
        
        print("\n1️⃣  Comprehensive Tests")
        os.chdir(project_root)
        os.system(f"{sys.executable} scripts/run_comprehensive_test.py")
        
        input("\nPress Enter to continue to demo...")
        
        print("\n2️⃣  Full Pipeline Demo")
        os.system(f"{sys.executable} scripts/demo_full_pipeline.py")
    
    elif choice == '0':
        print("\nGoodbye! 👋\n")
        return 0
    
    print("\n" + "█"*70)
    print("✅ Done!\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
