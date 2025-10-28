from ultralytics import YOLO
import pandas as pd
import os
import yaml
import numpy as np

# Configuration
MODEL_PATH = r"/Users/adityadatta/Documents/programming/499-299-498r/cse-499-yolov8-model-performance-checking/best.pt"
DATASET_YAML = r"/Users/adityadatta/Documents/programming/499-299-498r/cse-499-yolov8-model-performance-checking/dataset/data.yaml"
CONF_THRESHOLD = 0.75
OUTPUT_CSV = r"/Users/adityadatta/Documents/programming/499-299-498r/cse-499-yolov8-model-performance-checking/75_metrics_per_class.csv"

# Classes you want to focus on (EXACT names from your dataset)
FOCUS_CLASSES = [
    'staining or visible changes without cavitation',
    'calculus',
    'visible changes with microcavitation',
    'visible changes with cavitation',
    'non-carious lesion'
]

print("="*60)
print("CONFIGURATION")
print("="*60)
print(f"Model: {MODEL_PATH}")
print(f"Dataset YAML: {DATASET_YAML}")
print(f"Confidence Threshold: {CONF_THRESHOLD}")
print(f"Output CSV: {OUTPUT_CSV}")
print(f"Focus Classes: {len(FOCUS_CLASSES)}")
for fc in FOCUS_CLASSES:
    print(f"  - {fc}")
print()

# Load YOLOv8 model
print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded\n")

# Evaluate model (segmentation)
val_output_dir = r"/Users/adityadatta/Documents/programming/499-299-498r/cse-499-yolov8-model-performance-checking/runs/segment/val"
os.makedirs(val_output_dir, exist_ok=True)

print("Running validation...")
results = model.val(data=DATASET_YAML, conf=CONF_THRESHOLD, save_json=False, save_hybrid=False)
print("✓ Validation complete\n")

# Count images and instances PER CLASS
with open(DATASET_YAML, "r") as f:
    data_cfg = yaml.safe_load(f)

val_images_dir = data_cfg.get("val", "")
print(f"Validation images directory: {val_images_dir}")
labels_val_dir = val_images_dir.replace("images", "labels")
print(f"Labels directory: {labels_val_dir}")
print()

def count_images_instances_per_class(label_path):
    """Count total images/instances AND per-class breakdown"""
    total_images = 0
    total_instances = 0
    class_instances = {}  # class_id -> instance count
    class_images = {}     # class_id -> image count
    
    if not os.path.exists(label_path):
        print(f"WARNING: Label path does not exist: {label_path}")
        return 0, 0, class_instances, class_images
    
    label_files = []
    for root, _, files in os.walk(label_path):
        for f in files:
            if f.endswith(".txt"):
                label_files.append(os.path.join(root, f))
    
    print(f"Found {len(label_files)} label files")
    
    for label_file in label_files:
        total_images += 1
        classes_in_image = set()  # Track unique classes in this image
        
        with open(label_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    total_instances += 1
                    class_id = int(line.split()[0])
                    
                    # Count instances
                    class_instances[class_id] = class_instances.get(class_id, 0) + 1
                    
                    # Track classes in this image
                    classes_in_image.add(class_id)
        
        # Count images for each class
        for class_id in classes_in_image:
            class_images[class_id] = class_images.get(class_id, 0) + 1
    
    return total_images, total_instances, class_instances, class_images

total_images, total_instances, class_instances, class_images = count_images_instances_per_class(labels_val_dir)
print(f"Total validation images: {total_images}")
print(f"Total instances: {total_instances}")
print()

# Extract per-class metrics
original_names = results.names
seg_metrics = results.seg

print("="*60)
print("DATASET CLASSES")
print("="*60)
print(f"Total classes in model: {len(original_names)}")
print()
print("All classes:")
for idx, name in sorted(original_names.items()):
    instances = class_instances.get(idx, 0)
    images = class_images.get(idx, 0)
    is_focus = "FOCUS" if name in FOCUS_CLASSES else ""
    print(f"  {idx:2d}: {name:45s} ({images:3d} images, {instances:4d} instances) {is_focus}")
print()

print("="*60)
print("EXTRACTING METRICS")
print("="*60)

# Initialize metrics dictionary
metrics_dict = {}

try:
    # Extract metric arrays
    precision_array = seg_metrics.p if hasattr(seg_metrics, 'p') else None
    recall_array = seg_metrics.r if hasattr(seg_metrics, 'r') else None
    
    # Get AP50
    ap50_array = None
    if hasattr(seg_metrics, 'ap50'):
        ap50_array = seg_metrics.ap50
    elif hasattr(seg_metrics, 'ap') and isinstance(seg_metrics.ap, np.ndarray):
        if len(seg_metrics.ap.shape) == 2:
            ap50_array = seg_metrics.ap[:, 0]
    
    # Get mAP (average across IoU thresholds)
    ap_array = None
    if hasattr(seg_metrics, 'ap'):
        ap_attr = seg_metrics.ap
        if isinstance(ap_attr, np.ndarray):
            if len(ap_attr.shape) == 2:
                ap_array = ap_attr.mean(axis=1)
            elif len(ap_attr.shape) == 1:
                ap_array = ap_attr
    
    print(f"Precision array shape: {precision_array.shape if precision_array is not None else 'None'}")
    print(f"Recall array shape: {recall_array.shape if recall_array is not None else 'None'}")
    print(f"AP50 array shape: {ap50_array.shape if ap50_array is not None else 'None'}")
    print(f"mAP array shape: {ap_array.shape if ap_array is not None else 'None'}")
    print()
    
    # Extract metrics for each class
    for cls_idx in original_names.keys():
        cls_name = original_names[cls_idx]
        
        p_val = float(precision_array[cls_idx]) if precision_array is not None and len(precision_array) > cls_idx else 0.0
        r_val = float(recall_array[cls_idx]) if recall_array is not None and len(recall_array) > cls_idx else 0.0
        map50_val = float(ap50_array[cls_idx]) if ap50_array is not None and len(ap50_array) > cls_idx else 0.0
        map_val = float(ap_array[cls_idx]) if ap_array is not None and len(ap_array) > cls_idx else 0.0
        
        metrics_dict[cls_idx] = {
            'name': cls_name,
            'precision': p_val,
            'recall': r_val,
            'map50': map50_val,
            'map': map_val,
            'images': class_images.get(cls_idx, 0),
            'instances': class_instances.get(cls_idx, 0)
        }
    
    print("Metrics extracted for all classes\n")
    
except Exception as e:
    print(f"ERROR extracting metrics: {e}")
    import traceback
    traceback.print_exc()
    print()

# Aggregate metrics
print("="*60)
print("AGGREGATING METRICS")
print("="*60)

focus_class_metrics = {}
other_class_metrics = {
    'Precision': [], 'Recall': [], 'mAP50': [], 'mAP50-95': []
}

for cls_idx, metrics in metrics_dict.items():
    cls_name = metrics['name']
    
    if cls_name in FOCUS_CLASSES:
        focus_class_metrics[cls_name] = {
            'Images': metrics['images'],
            'Instances': metrics['instances'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'mAP50': metrics['map50'],
            'mAP50-95': metrics['map']
        }
        print(f"Focus class '{cls_name}': {metrics['images']} images, {metrics['instances']} instances")
    else:
        other_class_metrics['Precision'].append(metrics['precision'])
        other_class_metrics['Recall'].append(metrics['recall'])
        other_class_metrics['mAP50'].append(metrics['map50'])
        other_class_metrics['mAP50-95'].append(metrics['map'])

print(f"Non-focus classes: {len(other_class_metrics['Precision'])} classes aggregated into 'all'")
print()

# Build final metrics list
final_metrics = []

# Add "all" row
if other_class_metrics['Precision']:
    averaged = {
        'Class': 'all',
        'Images': total_images,
        'Instances': total_instances,
        'Precision': np.mean(other_class_metrics['Precision']),
        'Recall': np.mean(other_class_metrics['Recall']),
        'mAP50': np.mean(other_class_metrics['mAP50']),
        'mAP50-95': np.mean(other_class_metrics['mAP50-95'])
    }
    final_metrics.append(averaged)
else:
    final_metrics.append({
        'Class': 'all',
        'Images': total_images,
        'Instances': total_instances,
        'Precision': 0.0,
        'Recall': 0.0,
        'mAP50': 0.0,
        'mAP50-95': 0.0
    })

# Add focus classes with their specific counts
for focus_class in FOCUS_CLASSES:
    if focus_class in focus_class_metrics:
        vals = focus_class_metrics[focus_class]
        row = {
            'Class': focus_class,
            'Images': vals['Images'],
            'Instances': vals['Instances'],
            'Precision': vals['Precision'],
            'Recall': vals['Recall'],
            'mAP50': vals['mAP50'],
            'mAP50-95': vals['mAP50-95']
        }
        final_metrics.append(row)
    else:
        print(f"⚠ WARNING: Focus class '{focus_class}' not found in dataset")
        row = {
            'Class': focus_class,
            'Images': 0,
            'Instances': 0,
            'Precision': 0.0,
            'Recall': 0.0,
            'mAP50': 0.0,
            'mAP50-95': 0.0
        }
        final_metrics.append(row)

# Save to CSV
df = pd.DataFrame(final_metrics)
df = df[['Class', 'Images', 'Instances', 'Precision', 'Recall', 'mAP50', 'mAP50-95']]

print("="*60)
print("FINAL METRICS TABLE")
print("="*60)
print(df.to_string(index=False))
print()

# Ensure output directory exists
output_dir = os.path.dirname(OUTPUT_CSV)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

# Save with error handling
print("="*60)
print("SAVING RESULTS")
print("="*60)
try:
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully saved metrics to:")
    print(f"  {OUTPUT_CSV}")
    print(f"File exists: {os.path.exists(OUTPUT_CSV)}")
    if os.path.exists(OUTPUT_CSV):
        print(f"File size: {os.path.getsize(OUTPUT_CSV)} bytes")
except Exception as e:
    print(f"ERROR saving CSV: {e}")
    print(f"Attempted path: {OUTPUT_CSV}")
    # Try saving to current directory as fallback
    fallback_path = "75_metrics_per_class.csv"
    try:
        df.to_csv(fallback_path, index=False)
        print(f"Saved to fallback location: {os.path.abspath(fallback_path)}")
    except Exception as e2:
        print(f"Fallback also failed: {e2}")

print()
print("="*60)
print("COMPLETE")
print("="*60)