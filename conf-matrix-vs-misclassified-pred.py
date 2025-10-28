from ultralytics import YOLO
import pandas as pd
import numpy as np
import os
import yaml

# Configuration
MODEL_PATH = r"/Users/adityadatta/Documents/programming/499-299-498r/cse-499-yolov8-model-performance-checking/best.pt"
DATASET_YAML = r"/Users/adityadatta/Documents/programming/499-299-498r/cse-499-yolov8-model-performance-checking/dataset/data.yaml"
OUTPUT_CSV = r"/Users/adityadatta/Documents/programming/499-299-498r/cse-499-yolov8-model-performance-checking/confusion_matrix.csv"

# Confidence thresholds to test
CONF_THRESHOLDS = [0.25, 0.5, 0.75]

# Focus classes (exact names from your dataset - FIXED: using exact names from data.yaml)
FOCUS_CLASSES = [
    'staining or visible changes without cavitation',
    'calculus',
    'visible changes with microcavitation',
    'visible changes with cavitation',
    'non-carious lesion'  # FIXED: Added with exact name from data.yaml (singular, not plural)
]

# Display names for the table
DISPLAY_NAMES = {
    'staining or visible changes without cavitation': 'Staining',
    'calculus': 'Calculus',
    'visible changes with microcavitation': 'Micro cavitation',
    'visible changes with cavitation': 'Cavitation',
    'non-carious lesion': 'Non-carious lesion'  # FIXED: Added display name
}

# Load model and dataset info
model = YOLO(MODEL_PATH)

with open(DATASET_YAML, "r") as f:
    data_cfg = yaml.safe_load(f)
    class_names = data_cfg.get('names', [])

print(f"Loaded {len(class_names)} classes from data.yaml")
print(f"Focus classes to track: {FOCUS_CLASSES}")

# Verify all focus classes exist in the dataset
for focus_class in FOCUS_CLASSES:
    if focus_class not in class_names:
        print(f"WARNING: '{focus_class}' not found in dataset classes!")
    else:
        print(f"✓ Found '{focus_class}' in dataset")

# Create class name to index mapping
name_to_idx = {}
if isinstance(class_names, dict):
    name_to_idx = {v: k for k, v in class_names.items()}
else:
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}

# Get validation image paths
val_images_dir = data_cfg.get("val", "")
labels_val_dir = val_images_dir.replace("images", "labels")

# Collect all validation image paths
val_image_paths = []
for root, _, files in os.walk(val_images_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            val_image_paths.append(os.path.join(root, f))

print(f"Found {len(val_image_paths)} validation images")

# Function to load ground truth labels
def load_ground_truth(image_path, labels_dir):
    """Load ground truth labels for an image"""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(
        labels_dir,
        os.path.relpath(os.path.dirname(image_path), val_images_dir),
        base_name + ".txt"
    )
    
    gt_classes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    gt_classes.append(int(parts[0]))
    
    return gt_classes

# Run predictions at different confidence thresholds
results_by_threshold = {}

for conf_threshold in CONF_THRESHOLDS:
    print(f"\n{'='*60}")
    print(f"Running predictions with confidence threshold: {conf_threshold}")
    print(f"{'='*60}")
    
    # Initialize counters for this threshold
    confusion_data = {}
    for focus_class in FOCUS_CLASSES:
        confusion_data[focus_class] = {
            'correct': 0,
            'misclassified': 0,
            'total_gt': 0  # Total ground truth instances
        }
    
    # Background class
    confusion_data['Background'] = {
        'correct': 0,
        'misclassified': 0,
        'total_gt': 0
    }
    
    # Process each validation image
    for img_idx, img_path in enumerate(val_image_paths):
        if (img_idx + 1) % 20 == 0:
            print(f"Processing image {img_idx + 1}/{len(val_image_paths)}...")
        
        # Get ground truth
        gt_classes = load_ground_truth(img_path, labels_val_dir)
        
        # Run prediction
        results = model.predict(img_path, conf=conf_threshold, verbose=False)
        
        # Get predicted classes (for segmentation, use masks)
        pred_classes = []
        if len(results) > 0:
            # Try to get classes from masks (segmentation) or boxes (detection)
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                pred_classes = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
            elif results[0].boxes is not None:
                pred_classes = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
        
        # Convert class indices to names
        gt_class_names = []
        for cls_idx in gt_classes:
            if isinstance(class_names, list):
                gt_class_names.append(class_names[cls_idx] if cls_idx < len(class_names) else '')
            else:
                gt_class_names.append(class_names.get(cls_idx, ''))
        
        pred_class_names = []
        for cls_idx in pred_classes:
            if isinstance(class_names, list):
                pred_class_names.append(class_names[cls_idx] if cls_idx < len(class_names) else '')
            else:
                pred_class_names.append(class_names.get(cls_idx, ''))
        
        # Debug: Print for first few images
        if img_idx < 3:
            print(f"\nImage {img_idx}: {os.path.basename(img_path)}")
            print(f"  GT classes: {gt_class_names}")
            print(f"  Pred classes: {pred_class_names}")
        
        # Count for focus classes - using instance-level counting
        for focus_class in FOCUS_CLASSES:
            gt_count = gt_class_names.count(focus_class)
            pred_count = pred_class_names.count(focus_class)
            
            if gt_count > 0:
                confusion_data[focus_class]['total_gt'] += gt_count
                
                # Correct detections: minimum of GT and predictions
                correct = min(gt_count, pred_count)
                confusion_data[focus_class]['correct'] += correct
                
                # Misclassified (False Negatives): GT instances that weren't predicted
                false_negatives = gt_count - correct
                confusion_data[focus_class]['misclassified'] += false_negatives
                
                if img_idx < 3 and gt_count > 0:
                    print(f"  {focus_class}: GT={gt_count}, Pred={pred_count}, Correct={correct}, FN={false_negatives}")
        
        # Background handling
        has_focus_gt = any(cls_name in FOCUS_CLASSES for cls_name in gt_class_names)
        has_focus_pred = any(cls_name in FOCUS_CLASSES for cls_name in pred_class_names)
        
        # Count as background if no focus classes in GT
        if not has_focus_gt:
            confusion_data['Background']['total_gt'] += 1
            # Correct if also no focus classes predicted
            if not has_focus_pred:
                confusion_data['Background']['correct'] += 1
            else:
                # Misclassified if predicted focus class when there shouldn't be one
                confusion_data['Background']['misclassified'] += 1
    
    results_by_threshold[conf_threshold] = confusion_data
    
    print(f"\nResults for threshold {conf_threshold}:")
    for class_name, data in confusion_data.items():
        print(f"  {class_name}: Correct={data['correct']}, Misclassified={data['misclassified']}, Total GT={data['total_gt']}")

# Create the table
print(f"\n{'='*60}")
print("Creating Confusion Matrix Table")
print(f"{'='*60}")

table_data = []

# Add header row
header = ['', 'Confidence Threshold'] + [str(th) for th in CONF_THRESHOLDS]
table_data.append(header)

# Background row
bg_correct = ['Background', 'Correct'] + [results_by_threshold[th]['Background']['correct'] for th in CONF_THRESHOLDS]
bg_misclass = ['', 'Misclassified'] + [results_by_threshold[th]['Background']['misclassified'] for th in CONF_THRESHOLDS]
table_data.append(bg_correct)
table_data.append(bg_misclass)

# Focus class rows
for focus_class in FOCUS_CLASSES:
    display_name = DISPLAY_NAMES.get(focus_class, focus_class)
    
    correct_row = [display_name, 'Correct'] + [results_by_threshold[th][focus_class]['correct'] for th in CONF_THRESHOLDS]
    misclass_row = ['', 'Misclassified'] + [results_by_threshold[th][focus_class]['misclassified'] for th in CONF_THRESHOLDS]
    
    table_data.append(correct_row)
    table_data.append(misclass_row)

# Create DataFrame
df = pd.DataFrame(table_data[1:], columns=table_data[0])

print("\n" + "="*60)
print("CONFUSION MATRIX TABLE")
print("="*60)
print(df.to_string(index=False))

# Save to CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved confusion matrix to {OUTPUT_CSV}")