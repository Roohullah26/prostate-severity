"""
Multi-sequence tumor size regression model.

This module defines a neural network that predicts tumor dimensions (width, height, depth)
from multi-sequence MRI input (T2, ADC, DWI). It outputs continuous values in millimeters
and can optionally predict severity grades.

Architecture:
- Backbone: ResNet18 (shared features)
- Heads:
  1. Size regressor: [width_mm, height_mm, depth_mm]
  2. Severity classifier: [T1, T2, T3, T4] probability
  3. Confidence: overall prediction confidence
"""

import torch
import torch.nn as nn
from torchvision import models


class TumorSizePredictor(nn.Module):
    """Multi-task model for tumor size prediction and severity classification.
    
    Input: (B, 3, 224, 224) - stacked T2/ADC/DWI or similar multi-sequence MRI
    Outputs:
      - size: (B, 3) - [width_mm, height_mm, depth_mm]
      - severity_logits: (B, 4) - logits for [T1, T2, T3, T4]
      - confidence: (B, 1) - prediction confidence [0, 1]
    """
    
    def __init__(self, pretrained=True, in_channels=3, dropout_rate=0.3):
        """Initialize the model.
        
        Args:
            pretrained: Use ImageNet-pretrained ResNet18 backbone
            in_channels: Input channels (3 for T2/ADC/DWI stack)
            dropout_rate: Dropout rate in heads (default 0.3)
        """
        super().__init__()
        
        # Load backbone
        use_pretrained = pretrained and in_channels == 3
        self.backbone = models.resnet18(pretrained=use_pretrained)
        
        # Adapt conv1 for multi-channel input
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Remove final classification layer
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Shared feature dimension
        self.feature_dim = feature_dim
        
        # === Size Regression Head ===
        self.size_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 3),  # [width_mm, height_mm, depth_mm]
        )
        
        # === Severity Classification Head (T1/T2/T3/T4) ===
        self.severity_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 4),  # [T1, T2, T3, T4] logits
        )
        
        # === Confidence Head ===
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
        
        # Initialize size head with small weights (expect outputs ~10-50 mm)
        self._init_size_head()
        
    def _init_size_head(self):
        """Initialize size head with values biased toward typical tumor sizes."""
        # Typical tumor sizes: 10-30 mm range
        # Initialize final layer biases to ~20 mm
        with torch.no_grad():
            final_layer = self.size_head[-1]
            if hasattr(final_layer, 'bias'):
                final_layer.bias.fill_(20.0)
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: (B, 3, 224, 224) - input MRI stack
            
        Returns:
            dict with keys:
                - size: (B, 3) - predicted [width_mm, height_mm, depth_mm]
                - severity_logits: (B, 4) - logits for severity classification
                - severity_probs: (B, 4) - softmax probabilities
                - confidence: (B, 1) - prediction confidence
        """
        # Backbone feature extraction
        features = self.backbone(x)  # (B, 512)
        
        # Head predictions
        size = self.size_head(features)  # (B, 3)
        severity_logits = self.severity_head(features)  # (B, 4)
        severity_probs = torch.softmax(severity_logits, dim=1)  # (B, 4)
        confidence = self.confidence_head(features)  # (B, 1)
        
        return {
            'size': size,
            'severity_logits': severity_logits,
            'severity_probs': severity_probs,
            'confidence': confidence,
        }


class SizeRegressionLoss(nn.Module):
    """Combined loss for size regression and severity classification.
    
    Loss = α * MSE(size) + β * CrossEntropy(severity) + γ * confidence_loss
    """
    
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict from model.forward()
                - size: (B, 3)
                - severity_logits: (B, 4)
                - confidence: (B, 1)
            targets: dict with keys
                - size: (B, 3) ground truth sizes in mm
                - severity: (B,) ground truth labels [0=T1, 1=T2, 2=T3, 3=T4]
                - uncertainty: (B, 1) optional target confidence
                
        Returns:
            total_loss: scalar tensor
            loss_dict: dict with individual loss components for logging
        """
        size_loss = self.mse_loss(predictions['size'], targets['size'])
        
        severity_loss = self.ce_loss(
            predictions['severity_logits'], 
            targets['severity'].long()
        )
        
        # Optional: confidence loss (encourage model to be confident on easy samples)
        confidence_loss = 0.0
        if 'uncertainty' in targets:
            confidence_loss = self.bce_loss(
                predictions['confidence'],
                targets['uncertainty']
            )
        
        total_loss = (
            self.alpha * size_loss +
            self.beta * severity_loss +
            self.gamma * confidence_loss
        )
        
        return total_loss, {
            'total': total_loss.item(),
            'size': size_loss.item(),
            'severity': severity_loss.item(),
            'confidence': confidence_loss if isinstance(confidence_loss, float) else confidence_loss.item(),
        }


def make_size_predictor(pretrained=True, in_channels=3, checkpoint_path=None):
    """Factory function to create and optionally load a TumorSizePredictor.
    
    Args:
        pretrained: Use ImageNet-pretrained backbone
        in_channels: Input channels (3 for T2/ADC/DWI)
        checkpoint_path: Optional path to load saved model weights
        
    Returns:
        TumorSizePredictor instance
    """
    model = TumorSizePredictor(pretrained=pretrained, in_channels=in_channels)
    
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
    
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TumorSizePredictor(pretrained=False, in_channels=3)
    model = model.to(device)
    
    # Create dummy input (batch of 2)
    x = torch.randn(2, 3, 224, 224).to(device)
    
    # Forward pass
    output = model(x)
    
    print("Model output structure:")
    for key, val in output.items():
        print(f"  {key}: {val.shape}")
    
    # Test loss
    loss_fn = SizeRegressionLoss()
    targets = {
        'size': torch.randn(2, 3).to(device) * 20 + 15,  # Typical sizes: 10-50 mm
        'severity': torch.tensor([0, 2]).to(device),      # T1, T3
    }
    
    loss, loss_dict = loss_fn(output, targets)
    print(f"\nLoss: {loss.item():.4f}")
    print("Loss breakdown:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.4f}")
