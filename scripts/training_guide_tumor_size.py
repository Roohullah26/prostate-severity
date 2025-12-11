#!/usr/bin/env python3
"""
Training Guide for Tumor Size Prediction Model
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from size_predictor_model import TumorSizePredictorMultiSeq
import numpy as np


def print_guide():
    """Print comprehensive training guide"""
    
    guide = """
╔════════════════════════════════════════════════════════════════════════════╗
║           TUMOR SIZE PREDICTION MODEL - TRAINING GUIDE                    ║
╚════════════════════════════════════════════════════════════════════════════╝

📋 OVERVIEW
-----------
The tumor size prediction model uses multi-sequence MRI (T2, ADC, DWI) to:
  • Predict tumor size in millimeters (mm)
  • Generate bounding boxes
  • Classify severity (T1a, T1b, T1c, T2, T3+)

🔧 MODEL ARCHITECTURE
---------------------
Input:  3-channel image stack (T2, ADC, DWI) - Shape: (3, H, W)
Output: Tumor size in mm (continuous value)

Backbone:
  • ResNet-50 feature extraction
  • Multi-scale attention mechanisms
  • Feature fusion from all 3 sequences
  • Dense regression head

📊 TRAINING DATA REQUIREMENTS
-----------------------------
For optimal performance, prepare:

1. Image Pairs:
   • T2-weighted images (anatomical structure)
   • ADC maps (diffusion - tumor detection)
   • DWI images (diffusion - tumor differentiation)
   
2. Ground Truth Labels:
   • Manually measured tumor size (mm)
   • Bounding box coordinates (optional)
   • Clinical severity scores

3. Data Format:
   • Images: PNG, JPEG, or DICOM
   • Size: Recommended 256x256 or larger
   • Intensity: 0-255 or 0-1 (normalized)
   • Labels: CSV with columns: [image_id, t2_path, adc_path, dwi_path, tumor_size_mm]

📁 DATA DIRECTORY STRUCTURE
---------------------------
data/
├── train/
│   ├── t2/
│   │   ├── case_001.png
│   │   ├── case_002.png
│   │   └── ...
│   ├── adc/
│   │   ├── case_001.png
│   │   ├── case_002.png
│   │   └── ...
│   ├── dwi/
│   │   ├── case_001.png
│   │   ├── case_002.png
│   │   └── ...
│   └── labels.csv
├── val/
│   ├── t2/, adc/, dwi/
│   └── labels.csv
└── test/
    ├── t2/, adc/, dwi/
    └── labels.csv

🏋️ TRAINING PROCEDURE
---------------------
1. Prepare Data:
   python scripts/prepare_size_training_data.py \\
       --input-dir /path/to/raw/data \\
       --output-dir data/ \\
       --train-split 0.7 \\
       --val-split 0.15

2. Train Model:
   python src/train_size_model.py \\
       --data-dir data/ \\
       --epochs 100 \\
       --batch-size 32 \\
       --learning-rate 1e-4 \\
       --save-dir models/size_predictor/

3. Evaluate Model:
   python scripts/evaluate_size_model.py \\
       --model-path models/size_predictor/best_model.pth \\
       --data-dir data/test/

⚙️ TRAINING HYPERPARAMETERS
----------------------------
Learning Rate:      1e-4 to 1e-3
Batch Size:         16, 32, or 64
Epochs:             50-150
Optimizer:          Adam
Loss Function:      MSE + MAE hybrid
Regularization:     L2 (weight decay: 1e-4)
Early Stopping:     patience=15

🎯 EXPECTED PERFORMANCE
-----------------------
With good quality data (100+ cases):
  • MAE (Mean Absolute Error):   ±2-3 mm
  • RMSE:                         3-5 mm
  • Correlation (R²):            0.85-0.95

With limited data (<50 cases):
  • Use transfer learning from ImageNet
  • Augmentation: rotation, flip, elastic deformation
  • Consider pre-training on public prostate datasets

🔍 VALIDATION STRATEGY
----------------------
1. K-Fold Cross-Validation:
   python scripts/kfold_train_size_model.py --folds 5

2. Temporal Validation:
   Train on older cases, validate on newer cases

3. Clinical Validation:
   Compare predictions vs radiologist measurements

📈 METRICS TO TRACK
-------------------
During Training:
  • Training Loss (MSE)
  • Validation Loss
  • MAE (Mean Absolute Error)
  • RMSE (Root Mean Squared Error)

Per Patient:
  • Prediction accuracy within ±2mm
  • Bounding box IoU (Intersection over Union)
  • Severity classification accuracy

💾 MODEL CHECKPOINTING
---------------------
Save checkpoints:
  • Every 5 epochs
  • When validation loss improves
  • Best model (highest R²)
  
Example checkpoint files:
  models/size_predictor/checkpoint_epoch_05.pth
  models/size_predictor/best_model.pth

🔄 TRANSFER LEARNING
--------------------
If training data is limited:

1. Use pre-trained ResNet-50:
   model = TumorSizePredictorMultiSeq(pretrained=True)

2. Freeze early layers:
   for param in model.backbone.layer1.parameters():
       param.requires_grad = False
   for param in model.backbone.layer2.parameters():
       param.requires_grad = False

3. Fine-tune with lower learning rate:
   lr = 1e-5  # Lower than full training

📊 DATA AUGMENTATION
-------------------
Recommended augmentations:
  • Rotation:       ±15 degrees
  • Flip:           Horizontal (not vertical - anatomy matters)
  • Elastic:        σ=30, α=100
  • Noise:          Gaussian, σ=0.01-0.05
  • Brightness:     ±10%
  • Contrast:       0.9-1.1x

🚀 INFERENCE
-----------
After training:

1. Single Case:
   python scripts/run_tumor_size_pipeline.py t2.png adc.png dwi.png

2. Batch Inference:
   python scripts/batch_predict_tumor_size.py --data-dir data/test/

3. API Inference:
   python -m uvicorn webapp.fastapi_server:app --reload
   # Then use API client

📝 TROUBLESHOOTING
-----------------
Problem: High training loss, low validation accuracy
Solution: 
  • Increase data augmentation
  • Use transfer learning
  • Reduce learning rate
  • Add dropout/regularization

Problem: Overfitting (low training loss, high validation loss)
Solution:
  • Reduce model capacity
  • Increase L2 regularization
  • Use early stopping
  • More data or aggressive augmentation

Problem: Poor generalization to new data
Solution:
  • Validate on separate clinical sites
  • Include diverse patient populations
  • Use domain adaptation techniques
  • Cross-validate with radiologists

✅ CHECKLIST FOR TRAINING
------------------------
□ Data preparation complete
□ Train/val/test split (70/15/15)
□ Images normalized and resized
□ Labels verified with radiologist
□ Model architecture chosen
□ Hyperparameters tuned
□ Training started with monitoring
□ Validation loss improving
□ No data leakage between splits
□ Model evaluation metrics acceptable
□ Model saved and versioned
□ Performance on test set validated

🔗 RELATED SCRIPTS
------------------
• src/train_size_model.py          - Main training script
• src/size_predictor_model.py      - Model definition
• scripts/run_tumor_size_pipeline.py - Inference pipeline
• scripts/api_client_tumor_size.py  - API client
• webapp/fastapi_server.py         - API server

📚 REFERENCES
-------------
• ResNet: He et al., "Deep Residual Learning..." (2015)
• Multi-scale learning: Inception architecture
• MRI analysis: Prostate cancer detection literature
• Size prediction: Radiomics and clinical measurements

═════════════════════════════════════════════════════════════════════════════
"""
    print(guide)


def show_model_summary():
    """Show model architecture summary"""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*80 + "\n")
    
    model = TumorSizePredictorMultiSeq()
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")


def main():
    """Main entry point"""
    print_guide()
    
    try:
        show_model_summary()
    except Exception as e:
        print(f"\n[Info] Model summary unavailable: {e}")
    
    print("\n" + "="*80)
    print("For more information, see: TUMOR_SIZE_COMPLETE_GUIDE.md")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
