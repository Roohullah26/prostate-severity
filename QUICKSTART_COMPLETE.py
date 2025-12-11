"""
QUICK START GUIDE - Tumor Size Prediction System
Run this to see everything working in one place
"""
import sys
from pathlib import Path
import subprocess
import json

def print_banner(title):
    """Print formatted banner"""
    width = 70
    print("\n" + "╔" + "═" * (width - 2) + "╗")
    print("║" + title.center(width - 2) + "║")
    print("╚" + "═" * (width - 2) + "╝")


def print_section(title):
    """Print section header"""
    print(f"\n{'─' * 70}")
    print(f"► {title}")
    print(f"{'─' * 70}")


def option_1_run_demo():
    """Option 1: Run end-to-end demo"""
    print_section("Option 1: End-to-End Demo")
    print("""
This demo shows the complete pipeline:
1. Create synthetic MRI images (T2, ADC, DWI)
2. Extract tumor features from all sequences
3. Estimate tumor size
4. Classify TNM severity
5. Detect bounding box
6. Generate comprehensive report
    """)
    
    input("Press Enter to run demo...")
    try:
        subprocess.run([sys.executable, "scripts/end_to_end_demo.py"], check=True)
    except Exception as e:
        print(f"Error running demo: {e}")


def option_2_batch_analysis():
    """Option 2: Run batch analysis"""
    print_section("Option 2: Batch Processing")
    print("""
This processes multiple patient cases in batch:
1. Create 8 synthetic patient cases
2. Analyze each case
3. Generate summary statistics
4. Save results to CSV, JSON, and TXT
    """)
    
    input("Press Enter to run batch analysis...")
    try:
        subprocess.run([sys.executable, "scripts/batch_tumor_analyzer.py"], check=True)
    except Exception as e:
        print(f"Error running batch analysis: {e}")


def option_3_system_tests():
    """Option 3: Run system tests"""
    print_section("Option 3: Comprehensive System Tests")
    print("""
Tests all system components:
✓ NumPy image operations
✓ Tumor size estimation
✓ TNM classification
✓ Bounding box detection
✓ Multi-sequence integration
✓ JSON serialization
✓ Batch processing logic
✓ Performance testing
    """)
    
    input("Press Enter to run tests...")
    try:
        subprocess.run([sys.executable, "scripts/comprehensive_system_test.py"], check=True)
    except Exception as e:
        print(f"Error running tests: {e}")


def option_4_api_server():
    """Option 4: Start API server"""
    print_section("Option 4: Start FastAPI Server")
    print("""
This starts the FastAPI server with:
✓ /predict-size - Tumor size prediction endpoint
✓ /status - System status
✓ /health - Health check
✓ Interactive API documentation at http://localhost:8000/docs

Press Ctrl+C to stop the server
    """)
    
    input("Press Enter to start server (Ctrl+C to stop)...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "webapp.fastapi_server:app", 
            "--reload",
            "--host", "0.0.0.0",
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Error starting server: {e}")


def option_5_streamlit_ui():
    """Option 5: Start Streamlit UI"""
    print_section("Option 5: Start Streamlit Web UI")
    print("""
This starts the interactive Streamlit app with:
✓ Demo mode with interactive slider
✓ Synthetic data generator
✓ Image upload functionality
✓ Real-time predictions and visualizations

Streamlit will open at http://localhost:8501
    """)
    
    input("Press Enter to start Streamlit (Ctrl+C to stop)...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", 
            "run", "webapp/streamlit_app.py"
        ])
    except KeyboardInterrupt:
        print("\nStreamlit stopped")
    except Exception as e:
        print(f"Error starting Streamlit: {e}")


def option_6_api_client_demo():
    """Option 6: Show API client usage"""
    print_section("Option 6: API Client Demo")
    print("""
Example Python code for using the API client:

```python
from scripts.complete_api_client import TumorSizeAPIClient
import numpy as np

# Create client
client = TumorSizeAPIClient("http://localhost:8000")

# Load or create images
t2 = np.random.rand(256, 256)
adc = np.random.rand(256, 256)
dwi = np.random.rand(256, 256)

# Get prediction
result = client.predict_tumor_size(t2, adc, dwi, patient_id="P001")

# Print results
print(f"Tumor Size: {result['tumor_size_mm']} mm")
print(f"Severity: {result['severity']}")
print(f"Confidence: {result['confidence']}")
print(f"Bounding Box: {result['bounding_box']}")
```
    """)
    
    print("\nAPI client code is ready at: scripts/complete_api_client.py")


def option_7_view_documentation():
    """Option 7: View documentation"""
    print_section("Option 7: View Documentation")
    
    docs = {
        "README.md": "Main project documentation",
        "IMPLEMENTATION_GUIDE_TUMOR_SIZE.md": "Implementation details",
        "TUMOR_SIZE_COMPLETE_GUIDE.md": "Complete system guide",
        "QUICK_REFERENCE.md": "Quick reference",
    }
    
    print("\nAvailable documentation:\n")
    for i, (filename, description) in enumerate(docs.items(), 1):
        print(f"{i}. {filename:<40} - {description}")
    
    print("\nView these files in your editor for comprehensive documentation.")


def print_help():
    """Print system overview and help"""
    print_banner("TUMOR SIZE PREDICTION & ANALYSIS SYSTEM")
    
    print("""
This system provides end-to-end tumor size prediction with:

📊 INPUT:
  • T2-weighted MRI images
  • ADC maps
  • DWI sequences

🔮 PREDICTION:
  • Tumor size (mm)
  • TNM severity classification (T1-T4)
  • Confidence score
  • Bounding box detection

📈 VISUALIZATION & REPORTING:
  • Interactive web UI (Streamlit)
  • REST API (FastAPI)
  • Batch processing
  • Comprehensive reporting
    """)
    
    print_section("Available Options")
    print("""
1. Run end-to-end demo
2. Run batch analysis
3. Run system tests
4. Start FastAPI server
5. Start Streamlit UI
6. Show API client usage
7. View documentation
8. Exit
    """)


def main():
    """Main menu"""
    while True:
        print_help()
        
        choice = input("Select an option (1-8): ").strip()
        
        if choice == "1":
            option_1_run_demo()
        elif choice == "2":
            option_2_batch_analysis()
        elif choice == "3":
            option_3_system_tests()
        elif choice == "4":
            option_4_api_server()
        elif choice == "5":
            option_5_streamlit_ui()
        elif choice == "6":
            option_6_api_client_demo()
        elif choice == "7":
            option_7_view_documentation()
        elif choice == "8":
            print("\nGoodbye!")
            break
        else:
            print("Invalid option. Please select 1-8.")


if __name__ == "__main__":
    main()
