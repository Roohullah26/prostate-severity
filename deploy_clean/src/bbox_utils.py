"""
Bounding box detection and visualization for tumor predictions.

This module handles:
1. Converting predicted tumor dimensions to bounding boxes (rectangular and circular)
2. Visualizing predictions on MRI images
3. Computing intersection-over-union (IoU) for evaluation
4. Drawing predictions with metadata
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont


@dataclass
class TumorPrediction:
    """Container for tumor prediction results."""
    
    # Dimensions in mm
    width_mm: float
    height_mm: float
    depth_mm: float
    
    # Severity
    severity: str  # 'T1', 'T2', 'T3', 'T4'
    severity_logits: np.ndarray  # (4,) softmax probabilities
    
    # Confidence
    confidence: float
    
    # Center location (in pixel coordinates, relative to input image)
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    
    # Original image info
    image_size: Optional[Tuple[int, int]] = None  # (H, W)
    pixel_spacing_mm: Optional[Tuple[float, float]] = None  # (row_mm, col_mm)


class BoundingBoxGenerator:
    """Generate bounding boxes from tumor predictions."""
    
    # Severity thresholds (mm) - Customize based on clinical guidelines
    SEVERITY_THRESHOLDS = {
        'T1': (0, 20),      # ≤ 20 mm
        'T2': (20, 40),     # 20-40 mm
        'T3': (40, 60),     # 40-60 mm
        'T4': (60, 1000),   # > 60 mm
    }
    
    def __init__(self, pixel_spacing_mm=None):
        """Initialize generator.
        
        Args:
            pixel_spacing_mm: (row_spacing, col_spacing) in mm for DICOM PixelSpacing
        """
        self.pixel_spacing_mm = pixel_spacing_mm or (1.0, 1.0)
    
    @staticmethod
    def classify_severity(max_size_mm: float) -> str:
        """Classify tumor severity based on size.
        
        Args:
            max_size_mm: Maximum dimension of tumor in mm
            
        Returns:
            Severity grade: 'T1', 'T2', 'T3', or 'T4'
        """
        thresholds = [(20, 'T1'), (40, 'T2'), (60, 'T3'), (1000, 'T4')]
        for thresh, grade in thresholds:
            if max_size_mm <= thresh:
                return grade
        return 'T4'
    
    def size_mm_to_pixels(self, size_mm: float) -> int:
        """Convert size in mm to pixels using average pixel spacing.
        
        Args:
            size_mm: Size in millimeters
            
        Returns:
            Size in pixels (integer)
        """
        avg_spacing = np.mean(self.pixel_spacing_mm)
        return max(1, int(np.round(size_mm / avg_spacing)))
    
    def get_rectangular_bbox(self, prediction: TumorPrediction, 
                            center_x: Optional[float] = None,
                            center_y: Optional[float] = None) -> Dict:
        """Generate rectangular bounding box from prediction.
        
        Args:
            prediction: TumorPrediction object
            center_x: Center X coordinate in pixels (default: image center)
            center_y: Center Y coordinate in pixels (default: image center)
            
        Returns:
            dict with keys:
                - x1, y1, x2, y2: Bounding box coordinates (pixels)
                - width_px, height_px: Box dimensions in pixels
                - center_x, center_y: Center in pixels
                - bbox_type: 'rect'
        """
        if center_x is None or center_y is None:
            if prediction.center_x is not None and prediction.center_y is not None:
                center_x = prediction.center_x
                center_y = prediction.center_y
            else:
                # Default to image center
                if prediction.image_size:
                    center_y = prediction.image_size[0] / 2
                    center_x = prediction.image_size[1] / 2
                else:
                    center_x, center_y = 112, 112  # Assume 224x224
        
        width_px = self.size_mm_to_pixels(prediction.width_mm)
        height_px = self.size_mm_to_pixels(prediction.height_mm)
        
        x1 = int(center_x - width_px / 2)
        y1 = int(center_y - height_px / 2)
        x2 = int(center_x + width_px / 2)
        y2 = int(center_y + height_px / 2)
        
        return {
            'type': 'rect',
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'center_x': int(center_x),
            'center_y': int(center_y),
            'width_px': width_px,
            'height_px': height_px,
        }
    
    def get_circular_bbox(self, prediction: TumorPrediction,
                         center_x: Optional[float] = None,
                         center_y: Optional[float] = None) -> Dict:
        """Generate circular bounding box from prediction.
        
        Uses the maximum dimension as the diameter.
        
        Args:
            prediction: TumorPrediction object
            center_x: Center X coordinate in pixels (default: image center)
            center_y: Center Y coordinate in pixels (default: image center)
            
        Returns:
            dict with keys:
                - center_x, center_y: Circle center in pixels
                - radius_px: Radius in pixels
                - radius_mm: Radius in mm
                - diameter_mm: Diameter in mm
                - bbox_type: 'circle'
        """
        if center_x is None or center_y is None:
            if prediction.center_x is not None and prediction.center_y is not None:
                center_x = prediction.center_x
                center_y = prediction.center_y
            else:
                if prediction.image_size:
                    center_y = prediction.image_size[0] / 2
                    center_x = prediction.image_size[1] / 2
                else:
                    center_x, center_y = 112, 112
        
        # Use max dimension for diameter
        diameter_mm = max(prediction.width_mm, prediction.height_mm, prediction.depth_mm)
        radius_mm = diameter_mm / 2
        radius_px = self.size_mm_to_pixels(radius_mm)
        
        return {
            'type': 'circle',
            'center_x': int(center_x),
            'center_y': int(center_y),
            'radius_px': radius_px,
            'radius_mm': radius_mm,
            'diameter_mm': diameter_mm,
        }
    
    def compute_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """Compute Intersection over Union (IoU) between two rectangular bboxes.
        
        Args:
            bbox1, bbox2: Dicts with keys x1, y1, x2, y2
            
        Returns:
            IoU value [0, 1]
        """
        x1_inter = max(bbox1['x1'], bbox2['x1'])
        y1_inter = max(bbox1['y1'], bbox2['y1'])
        x2_inter = min(bbox1['x2'], bbox2['x2'])
        y2_inter = min(bbox1['y2'], bbox2['y2'])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        box1_area = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
        box2_area = (bbox2['x2'] - bbox2['x1']) * (bbox2['y2'] - bbox2['y1'])
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


class VisualizationHelper:
    """Helper for visualizing tumor predictions on images."""
    
    # Color scheme (BGR for OpenCV, RGB for PIL)
    COLORS_BGR = {
        'T1': (0, 255, 0),      # Green - small
        'T2': (0, 255, 255),    # Yellow - medium
        'T3': (0, 165, 255),    # Orange - large
        'T4': (0, 0, 255),      # Red - very large
    }
    
    COLORS_RGB = {
        'T1': (0, 255, 0),      # Green
        'T2': (255, 255, 0),    # Yellow
        'T3': (255, 165, 0),    # Orange
        'T4': (255, 0, 0),      # Red
    }
    
    @staticmethod
    def draw_circular_bbox_cv2(image: np.ndarray, bbox: Dict, 
                               prediction: TumorPrediction,
                               thickness: int = 2) -> np.ndarray:
        """Draw circular bounding box on image using OpenCV.
        
        Args:
            image: Input image (H, W, C) in BGR format
            bbox: Dict from get_circular_bbox()
            prediction: TumorPrediction object
            thickness: Line thickness in pixels
            
        Returns:
            Image with drawn circle
        """
        img = image.copy()
        color = VisualizationHelper.COLORS_BGR.get(prediction.severity, (200, 200, 200))
        
        center = (bbox['center_x'], bbox['center_y'])
        radius = bbox['radius_px']
        
        # Draw circle
        cv2.circle(img, center, radius, color, thickness)
        
        # Draw center point
        cv2.circle(img, center, 3, color, -1)
        
        # Add text label
        text = f"{prediction.severity} {prediction.width_mm:.1f}mm (conf:{prediction.confidence:.2f})"
        cv2.putText(img, text, (bbox['center_x'] - 50, bbox['center_y'] - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img
    
    @staticmethod
    def draw_rectangular_bbox_cv2(image: np.ndarray, bbox: Dict,
                                 prediction: TumorPrediction,
                                 thickness: int = 2) -> np.ndarray:
        """Draw rectangular bounding box on image using OpenCV.
        
        Args:
            image: Input image (H, W, C) in BGR format
            bbox: Dict from get_rectangular_bbox()
            prediction: TumorPrediction object
            thickness: Line thickness in pixels
            
        Returns:
            Image with drawn rectangle
        """
        img = image.copy()
        color = VisualizationHelper.COLORS_BGR.get(prediction.severity, (200, 200, 200))
        
        pt1 = (bbox['x1'], bbox['y1'])
        pt2 = (bbox['x2'], bbox['y2'])
        
        # Draw rectangle
        cv2.rectangle(img, pt1, pt2, color, thickness)
        
        # Add text label
        text = f"{prediction.severity} {prediction.width_mm:.1f}x{prediction.height_mm:.1f}mm"
        cv2.putText(img, text, (bbox['x1'], max(10, bbox['y1'] - 5)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img
    
    @staticmethod
    def draw_circular_bbox_pil(image: Image.Image, bbox: Dict,
                               prediction: TumorPrediction,
                               thickness: int = 2) -> Image.Image:
        """Draw circular bounding box on PIL image.
        
        Args:
            image: PIL Image
            bbox: Dict from get_circular_bbox()
            prediction: TumorPrediction object
            thickness: Line thickness in pixels
            
        Returns:
            PIL Image with drawn circle
        """
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        color = VisualizationHelper.COLORS_RGB.get(prediction.severity, (200, 200, 200))
        
        x, y = bbox['center_x'], bbox['center_y']
        r = bbox['radius_px']
        
        # Draw circle
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=thickness)
        
        # Draw center
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
        
        # Add text
        text = f"{prediction.severity} {prediction.width_mm:.1f}mm"
        try:
            draw.text((x - 40, y - r - 15), text, fill=color)
        except:
            # Fallback if font unavailable
            pass
        
        return img
    
    @staticmethod
    def create_prediction_summary(prediction: TumorPrediction) -> str:
        """Create a text summary of prediction.
        
        Args:
            prediction: TumorPrediction object
            
        Returns:
            Formatted text summary
        """
        summary = f"""
=== Tumor Prediction Summary ===
Severity: {prediction.severity}
Dimensions: {prediction.width_mm:.2f} x {prediction.height_mm:.2f} x {prediction.depth_mm:.2f} mm
Confidence: {prediction.confidence:.3f}
Severity probabilities:
  T1 (≤20mm): {prediction.severity_logits[0]:.3f}
  T2 (20-40mm): {prediction.severity_logits[1]:.3f}
  T3 (40-60mm): {prediction.severity_logits[2]:.3f}
  T4 (>60mm): {prediction.severity_logits[3]:.3f}
"""
        return summary


if __name__ == "__main__":
    # Test the bbox generator
    pred = TumorPrediction(
        width_mm=15.5,
        height_mm=18.2,
        depth_mm=14.1,
        severity='T2',
        severity_logits=np.array([0.1, 0.6, 0.2, 0.1]),
        confidence=0.92,
        image_size=(224, 224),
        pixel_spacing_mm=(1.0, 1.0)
    )
    
    gen = BoundingBoxGenerator(pixel_spacing_mm=(1.0, 1.0))
    
    # Test rectangular bbox
    rect_bbox = gen.get_rectangular_bbox(pred)
    print("Rectangular bbox:", rect_bbox)
    
    # Test circular bbox
    circ_bbox = gen.get_circular_bbox(pred)
    print("Circular bbox:", circ_bbox)
    
    # Test severity classification
    print(f"Severity for 25mm: {gen.classify_severity(25)}")
    print(f"Severity for 50mm: {gen.classify_severity(50)}")
    
    # Test summary
    print(VisualizationHelper.create_prediction_summary(pred))
