"""
Dental Image Detection and Annotation Application

This Streamlit application provides:
1. YOLOv8-based detection/segmentation of dental diseases
2. Interactive annotation tool for creating custom datasets
3. Gemini AI integration for disease classification and consultation
4. YOLO format label generation for training data

Author: Aditya Narayan Datta Mallick
"""

from pathlib import Path
import os
import io
import re
import uuid
import time
import shutil

import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

import numpy as np

import settings
import helper

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Dental Image Segmentation and Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styling with gradient background
page_bg_img = f"""
<style>
.stApp, .stSidebar {{
background: #dce3cf;
background: linear-gradient(358deg, rgba(220, 227, 207, 1) 0%, rgba(211, 227, 211, 1) 50%, rgba(151, 199, 194, 1) 100%);
}}

/* Remove white header bar */
header {{
background: transparent !important;
}}

.stApp > header {{
background-color: transparent !important;
}}

/* Remove top padding */
.block-container {{
padding-top: 2rem !important;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Dental Image Detection and Annotation Application")

# SESSION STATE INITIALIZATION
if "img_file" not in st.session_state:
    st.session_state.img_file = None
if "cropped_img" not in st.session_state:
    st.session_state.cropped_img = None
if "crop_rect" not in st.session_state:
    st.session_state.crop_rect = None
if "label" not in st.session_state:
    st.session_state.label = ""
if "gemini_label" not in st.session_state:
    st.session_state.gemini_label = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "original_uploaded_img" not in st.session_state:
    st.session_state.original_uploaded_img = None
if "original_image_filename" not in st.session_state:
    st.session_state.original_image_filename = None

# SIDEBAR: MODEL CONFIGURATION
st.sidebar.header("DL Model Configuration")

# Model type selection
model_type = st.sidebar.radio(
    "Choose Task", ["Detection with Segmentation Masks"])

# Confidence threshold slider
confidence = float(st.sidebar.slider(
    "Confidence Threshold (%)", 15, 100, 21)) / 100

# Load the selected model
model_path = Path(
    settings.DETECTION_MODEL
    if model_type == "Detection"
    else settings.SEGMENTATION_MODEL
)
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Error loading model from path: {model_path}")
    st.error(ex)

# SIDEBAR: IMAGE SOURCE
st.sidebar.header("Image Configuration")
source_radio = st.sidebar.radio("Select Your Source", settings.SOURCES_LIST)

# IMAGE UPLOAD AND DETECTION
source_img = None
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image.", type=("jpg", "jpeg", "png", "bmp", "webp")
    )
    col1, col2 = st.columns(2)

    # Left column: Display uploaded or default image
    with col1:
        try:
            if source_img is None:
                # Show default image if none uploaded
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = Image.open(default_image_path)
                st.image(
                    default_image, caption="Default Image", width=365
                )
            else:
                # Display uploaded image with progress bar
                uploaded_image = Image.open(source_img)
                progress_bar = st.progress(0)
                for percent_complete in range(0, 101, 10):
                    time.sleep(0.05)
                    progress_bar.progress(percent_complete)
                progress_bar.empty()
                st.session_state.img_file = uploaded_image
                st.session_state.original_uploaded_img = uploaded_image
                # Store original filename without extension
                st.session_state.original_image_filename = os.path.splitext(source_img.name)[
                    0]
                st.image(
                    source_img, caption="Uploaded Image Successfully", width=365)
        except Exception as ex:
            st.error(f"Error loading image: {ex}")

    # Right column: Display detection results
    with col2:
        if source_img is None:
            # Show default detected image
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            st.image(
                default_detected_image_path,
                caption="Detected Image",
                width=365,
            )
        else:
            # Run detection when button is clicked
            if st.sidebar.button("Detect Disease"):
                res = model.predict(uploaded_image, conf=confidence)
                res_plotted = res[0].plot(labels=True)[:, :, ::-1]
                st.image(
                    res_plotted, caption="Detected Image", width=365
                )
                # Save detected image for annotation
                st.session_state.img_file = Image.fromarray(res_plotted)
                st.session_state.img_file.save(os.path.join(settings.ROOT, "detected_image.jpg"))

# ANNOTATION TOOL
st.title("Annotation Tool")

# Cropper configuration
realtime_update = st.checkbox("Update in Real Time", value=True)
box_color = st.color_picker("Box Color", value="#0b4cd9")
aspect_choice = st.radio("Aspect Ratio", options=[
                         "1:1", "16:9", "4:3", "2:3", "Free"])
aspect_dict = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "Free": None,
}
aspect_ratio = aspect_dict[aspect_choice]

# Load detected image if available
detected_img_path = "detected_image.jpg"
if st.session_state.img_file is None:
    if os.path.exists(detected_img_path):
        st.session_state.img_file = Image.open(detected_img_path)
    else:
        st.error("Detected image not found.")
        st.stop()

# Get original image dimensions
img_file = st.session_state.img_file
img_width, img_height = img_file.size

# Display original predicted image
st.header("Predicted Image (Original Size)")
st.image(img_file, width=365)

# Interactive cropping tool
st.header("Edit Bounding Boxes")
crop_result = st_cropper(
    img_file,
    realtime_update=realtime_update,
    box_color=box_color,
    aspect_ratio=aspect_ratio,
    return_type="box",  # Returns dict with crop coordinates
)

# Store crop coordinates and create cropped image
if crop_result is not None:
    # Check if crop has changed (compare with previous crop)
    crop_changed = (
        st.session_state.crop_rect is None or
        crop_result["left"] != st.session_state.crop_rect["left"] or
        crop_result["top"] != st.session_state.crop_rect["top"] or
        crop_result["width"] != st.session_state.crop_rect["width"] or
        crop_result["height"] != st.session_state.crop_rect["height"]
    )

    st.session_state.crop_rect = crop_result
    # Manually crop using rectangle coordinates from original image
    left = crop_result["left"]
    top = crop_result["top"]
    width = crop_result["width"]
    height = crop_result["height"]
    st.session_state.cropped_img = img_file.crop(
        (left, top, left + width, top + height)
    )

    # Reset analysis when crop changes
    if crop_changed:
        st.session_state.analysis_done = False
        st.session_state.gemini_label = None
        st.session_state.chat_history = []  # Also clear chat history

# Display cropped preview
st.header("Cropped Image Preview:")
if st.session_state.cropped_img:
    st.image(st.session_state.cropped_img, width=300)

# Define the 44 dental classes as per dataset
DENTAL_CLASSES = [
    '11', '12', '13', '14', '15', '16', '17', '18',
    '21', '22', '23', '24', '25', '26', '27', '28',
    '31', '32', '33', '34', '35', '36', '37', '38',
    '41', '42', '43', '44', '45', '46', '47', '48',
    'amalgam filling', 'calculus', 'fixed prosthesis', 'incisive papilla',
    'non-carious lesion', 'palatine raphe', 'staining or visible changes without cavitation',
    'temporary restoration', 'tongue', 'tooth coloured filling',
    'visible changes with cavitation', 'visible changes with microcavitation'
]

# Label input
st.header("Enter Label for the bounding box:")

# Create label options with ID: Name format
label_options = [f"{i}: {cls}" for i, cls in enumerate(DENTAL_CLASSES)]

selected_label = st.selectbox(
    "Select class (0-43):",
    options=label_options,
    index=0
)

# Extract just the class ID from selection
st.session_state.label = selected_label.split(":")[0]
st.header(
    f"Selected Label: {st.session_state.label} - {selected_label.split(': ', 1)[1]}")

# UTILITY FUNCTIONS


def get_unique_filename():
    """Generate a unique filename using UUID."""
    return str(uuid.uuid4())


def save_img_label_yolo_format(img_width, img_height, crop_rect, label, cropped_img):
    """
    Save cropped image and append YOLO format label to existing label file.
    Uses original image filename to group all annotations together.

    YOLO format: <class> <x_center> <y_center> <width> <height>
    All coordinates are normalized (0-1) relative to original image dimensions.

    Args:
        img_width (int): Original image width in pixels
        img_height (int): Original image height in pixels
        crop_rect (dict): Dictionary with 'left', 'top', 'width', 'height' from st_cropper
        label (str): Class label for the bounding box
        cropped_img (PIL.Image): The cropped image to save
    """
    # Use original filename or fallback to UUID if not available
    if st.session_state.original_image_filename:
        base_filename = st.session_state.original_image_filename
    else:
        base_filename = get_unique_filename()
        st.warning("Original filename not found, using UUID instead.")

    # Extract absolute crop coordinates from original image
    x_min = crop_rect["left"]
    y_min = crop_rect["top"]
    x_max = x_min + crop_rect["width"]
    y_max = y_min + crop_rect["height"]

    # Convert to YOLO format (normalized center coordinates and dimensions)
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width_norm = crop_rect["width"] / img_width
    height_norm = crop_rect["height"] / img_height

    # Create output directory
    os.makedirs("custom-labels", exist_ok=True)

    # Convert RGBA to RGB if necessary
    if cropped_img.mode in ("RGBA", "LA"):
        cropped_img = cropped_img.convert("RGB")

    # APPEND to existing label file (not overwrite)
    label_path = os.path.join("custom-labels", f"{base_filename}.txt")
    with open(label_path, "a") as f:  # Changed from "w" to "a" for append
        f.write(
            f"{label} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n"
        )

    # Save cropped image with incremental suffix
    # Count existing crops for this image
    existing_crops = [f for f in os.listdir("custom-labels")
                      if f.startswith(base_filename) and f.endswith(".jpg")]
    crop_number = len(existing_crops)

    image_path = os.path.join(
        "custom-labels", f"{base_filename}_crop{crop_number}.jpg")
    cropped_img.save(image_path)

    st.success(
        f"Saved crop #{crop_number} and appended label:\n- {image_path}\n- {label_path}")
    st.info(
        f"YOLO coords: class={label}, x_center={x_center:.4f}, y_center={y_center:.4f}, width={width_norm:.4f}, height={height_norm:.4f}"
    )


def save_cropped_image(cropped_img):
    """Save only the cropped image without YOLO label."""
    filename = get_unique_filename()

    # Convert RGBA to RGB if necessary
    if cropped_img.mode in ("RGBA", "LA"):
        cropped_img = cropped_img.convert("RGB")

    os.makedirs("custom-labels", exist_ok=True)
    path = os.path.join("custom-labels", f"{filename}.jpg")
    cropped_img.save(path)
    st.success(f"Cropped image saved as {path}")


def pil_to_bytes(img: Image.Image) -> bytes:
    """
    Convert PIL Image to JPEG bytes for API transmission.
    Resizes large images to reduce payload size.

    Args:
        img (PIL.Image): Input image

    Returns:
        bytes: JPEG-encoded image data
    """
    max_size = (1024, 1024)
    img.thumbnail(max_size)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


def sanitize_class_name(name: str) -> str:
    """
    Clean and normalize class names for YOLO compatibility.
    - Converts to lowercase
    - Removes special characters
    - Limits to first 2 words
    - Replaces spaces with underscores

    Args:
        name (str): Raw class name from Gemini

    Returns:
        str: Sanitized class name safe for YOLO
    """
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_ ]", "", name)
    name = "_".join([t for t in name.split() if t][:2])
    return name if name else "unknown"


def get_latest_file(folder="custom-labels", ext_list=[".jpg", ".jpeg", ".png"]):
    """
    Get the most recently created image file from a folder.

    Args:
        folder (str): Folder name to search
        ext_list (list): List of valid image extensions

    Returns:
        str: Path to latest file, or None if folder is empty
    """
    folder_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), folder)
    if not os.path.exists(folder_path):
        return None

    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if any(f.lower().endswith(ext) for ext in ext_list)
    ]

    return max(files, key=os.path.getctime) if files else None


def save_gemini_label_and_image(latest_image_path, gemini_label):
    """
    Update YOLO label file with Gemini-predicted class while preserving bounding box.
    Copies both image and updated label to gemini-label folder.

    Args:
        latest_image_path (str): Path to the cropped image
        gemini_label (str): Predicted class name from Gemini
    """
    gemini_folder = "gemini-label"
    os.makedirs(gemini_folder, exist_ok=True)

    # Copy image to gemini folder
    image_filename = os.path.basename(latest_image_path)
    dest_image_path = os.path.join(gemini_folder, image_filename)
    shutil.copy2(latest_image_path, dest_image_path)

    # Update label file with Gemini class
    txt_file = os.path.splitext(latest_image_path)[0] + ".txt"
    save_txt_path = os.path.join(
        gemini_folder, os.path.splitext(image_filename)[0] + ".txt"
    )

    if os.path.exists(txt_file):
        # Read existing label
        with open(txt_file, "r") as f:
            lines = f.readlines()

        # Replace class ID while keeping bounding box coordinates
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                parts[0] = gemini_label  # Update class, preserve bbox
                new_lines.append(" ".join(parts))

        # Save updated label
        with open(save_txt_path, "w") as f:
            f.write("\n".join(new_lines) + "\n")

        st.success(
            f"Gemini-corrected label saved:\n{save_txt_path}\nand image copied to:\n{dest_image_path}"
        )
    else:
        st.error(
            f"No original YOLO label found for {latest_image_path}. "
            "Please first save the cropped image and label."
        )


def save_original_with_yolo_label(
    img_width, img_height, crop_rect, label, original_img
):
    """
    Save the original full-size image once and append YOLO label to its label file.
    This preserves the complete context while still marking the detected region.

    Args:
        img_width (int): Original image width in pixels
        img_height (int): Original image height in pixels
        crop_rect (dict): Dictionary with 'left', 'top', 'width', 'height' from st_cropper
        label (str): Class label for the bounding box
        original_img (PIL.Image): The original full-size image to save
    """
    # Use original filename or fallback to UUID
    if st.session_state.original_image_filename:
        base_filename = st.session_state.original_image_filename
    else:
        base_filename = get_unique_filename()
        st.warning("Original filename not found, using UUID instead.")

    # Extract absolute crop coordinates from original image
    x_min = crop_rect["left"]
    y_min = crop_rect["top"]
    x_max = x_min + crop_rect["width"]
    y_max = y_min + crop_rect["height"]

    # Convert to YOLO format (normalized center coordinates and dimensions)
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width_norm = crop_rect["width"] / img_width
    height_norm = crop_rect["height"] / img_height

    # Create output directory for original images inside ROOT
    output_dir = os.path.join(settings.ROOT, "original-labels")
    os.makedirs(output_dir, exist_ok=True)

    # Convert RGBA to RGB if necessary
    if original_img.mode in ("RGBA", "LA"):
        original_img = original_img.convert("RGB")

    # APPEND to existing label file
    label_path = os.path.join(output_dir, f"{base_filename}.txt")
    with open(label_path, "a") as f:  # append mode
        f.write(
            f"{label} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n"
        )

    # Save original image only if it doesn't exist
    image_path = os.path.join(output_dir, f"{base_filename}.jpg")
    if not os.path.exists(image_path):
        original_img.save(image_path)
        st.success(f"Saved original image: {image_path}")
    else:
        st.info(f"Original image already exists: {image_path}")

    # Count total annotations
    with open(label_path, "r") as f:
        num_annotations = len(f.readlines())

    st.success(
        f"Appended annotation to label file (total: {num_annotations}):\n- {label_path}")
    st.info(
        f"YOLO coords: class={label}, x_center={x_center:.4f}, y_center={y_center:.4f}, width={width_norm:.4f}, height={height_norm:.4f}"
    )


def save_label_only(img_width, img_height, crop_rect, label):
    """
    Save only the YOLO format label file without any image.

    Args:
        img_width (int): Original image width in pixels
        img_height (int): Original image height in pixels
        crop_rect (dict): Dictionary with 'left', 'top', 'width', 'height' from st_cropper
        label (str): Class label for the bounding box
    """
    filename = get_unique_filename()

    # Extract absolute crop coordinates from original image
    x_min = crop_rect["left"]
    y_min = crop_rect["top"]
    x_max = x_min + crop_rect["width"]
    y_max = y_min + crop_rect["height"]

    # Convert to YOLO format (normalized center coordinates and dimensions)
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width_norm = crop_rect["width"] / img_width
    height_norm = crop_rect["height"] / img_height

    # Create output directory
    os.makedirs("custom-labels", exist_ok=True)

    # Save YOLO format label file
    label_path = os.path.join("custom-labels", f"{filename}.txt")
    with open(label_path, "w") as f:
        f.write(
            f"{label} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n"
        )

    st.success(f"Saved label only:\n- {label_path}")
    st.info(
        f"YOLO coords: class={label}, x_center={x_center:.4f}, y_center={y_center:.4f}, width={width_norm:.4f}, height={height_norm:.4f}"
    )


# SAVE BUTTONS SECTION

if (
    st.session_state.label
    and st.session_state.cropped_img
    and st.session_state.crop_rect
):
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Save Cropped Image Only"):
            save_cropped_image(st.session_state.cropped_img)

    with col2:
        if st.button("Save Label Only"):
            save_label_only(
                img_width,
                img_height,
                st.session_state.crop_rect,
                st.session_state.label,
            )

    col3, col4 = st.columns(2)

    with col3:
        if st.button("Save YOLO Label + Cropped Image"):
            save_img_label_yolo_format(
                img_width,
                img_height,
                st.session_state.crop_rect,
                st.session_state.label,
                st.session_state.cropped_img,
            )

    with col4:
        if st.button("Save YOLO Label + Original Image"):
            # Use the original uploaded image, not the detected one
            if st.session_state.original_uploaded_img is not None:
                # Get dimensions of ORIGINAL image (not detected)
                orig_width, orig_height = st.session_state.original_uploaded_img.size

                # Need to recalculate crop_rect relative to original image
                # if the detected image has different dimensions
                detected_width, detected_height = st.session_state.img_file.size

                # Scale crop coordinates if image sizes differ
                if (detected_width != orig_width) or (detected_height != orig_height):
                    scale_x = orig_width / detected_width
                    scale_y = orig_height / detected_height

                    scaled_crop_rect = {
                        "left": int(st.session_state.crop_rect["left"] * scale_x),
                        "top": int(st.session_state.crop_rect["top"] * scale_y),
                        "width": int(st.session_state.crop_rect["width"] * scale_x),
                        "height": int(st.session_state.crop_rect["height"] * scale_y),
                    }
                else:
                    scaled_crop_rect = st.session_state.crop_rect

                save_original_with_yolo_label(
                    orig_width,  # Use original dimensions
                    orig_height,
                    scaled_crop_rect,  # Use scaled coordinates
                    st.session_state.label,
                    st.session_state.original_uploaded_img,  # Pass original image
                )
            else:
                st.error("Original image not found. Please upload an image first.")
else:
    if not st.session_state.label:
        st.error("Label cannot be empty.")
    elif not st.session_state.cropped_img:
        st.error("Please crop an area first.")

# IMPROVED GEMINI AI INTEGRATION
st.title("Gemini 2.5 Flash - Cropped Tooth Disease Detection")

# Load Gemini API key from environment
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in .env")
    st.stop()

# Configure Gemini with generation parameters for optimal performance
genai.configure(api_key=GEMINI_API_KEY)

# IMPROVED: Higher temperature for better classification variety
generation_config = genai.types.GenerationConfig(
    temperature=0.4,  # Increased from 0.1 to allow more variation
    top_p=0.95,
    top_k=64,  # Increased from 40
    max_output_tokens=200,  # Increased from 100 to allow reasoning
)

# Generation config for chat (more creative)
chat_generation_config = genai.types.GenerationConfig(
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    max_output_tokens=1024,
)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config=generation_config
)

# IMPROVED PROMPT: More detailed and structured
PROMPT_FILE = "gemini_prompt.txt"
IMPROVED_PROMPT = """You are an expert dental diagnostician analyzing a cropped dental image.

CLASSIFICATION TASK:
Carefully examine this dental image and identify what is shown. Then select the ONE most accurate class from this list:

TOOTH NUMBERS (0-31):
'11', '12', '13', '14', '15', '16', '17', '18' (Upper right quadrant)
'21', '22', '23', '24', '25', '26', '27', '28' (Upper left quadrant)
'31', '32', '33', '34', '35', '36', '37', '38' (Lower left quadrant)
'41', '42', '43', '44', '45', '46', '47', '48' (Lower right quadrant)

DENTAL CONDITIONS/FEATURES (32-43):
'amalgam filling' - Dark metallic silver/gray filling material
'calculus' - Hardened plaque deposits, usually yellow/brown
'fixed prosthesis' - Dental bridge or crown (artificial tooth structure)
'incisive papilla' - Small bump of tissue behind upper front teeth
'non-carious lesion' - Tooth damage not from decay (erosion, abrasion, etc.)
'palatine raphe' - Ridge line in center of palate roof
'staining or visible changes without cavitation' - Discoloration but no hole
'temporary restoration' - Temporary filling (often white/off-white)
'tongue' - Visible tongue tissue
'tooth coloured filling' - White/beige composite filling material
'visible changes with cavitation' - Decay with an actual hole/cavity
'visible changes with microcavitation' - Very small cavity starting to form

ANALYSIS STEPS:
1. Is this a tooth? If yes, which tooth number based on position?
2. Is this a filling? If yes, what type (amalgam=dark, tooth-colored=white)?
3. Is this a dental condition? Look for decay, calculus, staining, etc.
4. Is this soft tissue? (tongue, incisive papilla, palatine raphe)

OUTPUT FORMAT:
Respond with ONLY the class name from the list above. No explanation, no quotes, just the exact class name.

Examples:
- If you see a dark metallic filling → amalgam filling
- If you see yellow/brown deposits on tooth → calculus
- If you see a white/beige filling → tooth coloured filling
- If you see a natural tooth with no obvious pathology → appropriate tooth number (11-48)
- If you see decay with a hole → visible changes with cavitation

Return ONLY the class name."""

# Save improved prompt
with open(PROMPT_FILE, "w") as f:
    f.write(IMPROVED_PROMPT)


def map_class_to_id(class_name: str) -> int:
    """Map class name to YOLO class ID (0-43) with improved fuzzy matching."""
    class_name = class_name.strip().lower()

    # Try exact match first
    for idx, dental_class in enumerate(DENTAL_CLASSES):
        if class_name == dental_class.lower():
            return idx

    # Try partial match
    for idx, dental_class in enumerate(DENTAL_CLASSES):
        if class_name in dental_class.lower() or dental_class.lower() in class_name:
            return idx

    # Try number extraction for tooth numbers
    import re
    numbers = re.findall(r'\d+', class_name)
    if numbers:
        tooth_num = numbers[0]
        if tooth_num in DENTAL_CLASSES:
            return DENTAL_CLASSES.index(tooth_num)

    return -1  # Unknown class


def save_gemini_yolo_label(detected_img, crop_rect, class_name, original_img):
    """
    Append Gemini-predicted YOLO label to existing label file for original image.
    """
    # Map class name to ID
    class_id = map_class_to_id(class_name)

    if class_id == -1:
        st.error(
            f"Could not map '{class_name}' to any of the 44 classes. Using class 0 as fallback.")
        class_id = 0

    # Use original filename or fallback to UUID
    if st.session_state.original_image_filename:
        base_filename = st.session_state.original_image_filename
    else:
        base_filename = get_unique_filename()
        st.warning("Original filename not found, using UUID instead.")

    # Get dimensions of both images
    detected_width, detected_height = detected_img.size
    orig_width, orig_height = original_img.size

    # Scale crop coordinates from detected image to original image dimensions
    if (detected_width != orig_width) or (detected_height != orig_height):
        scale_x = orig_width / detected_width
        scale_y = orig_height / detected_height

        scaled_crop_rect = {
            "left": int(crop_rect["left"] * scale_x),
            "top": int(crop_rect["top"] * scale_y),
            "width": int(crop_rect["width"] * scale_x),
            "height": int(crop_rect["height"] * scale_y),
        }
    else:
        scaled_crop_rect = crop_rect

    # Extract absolute crop coordinates from ORIGINAL image
    x_min = scaled_crop_rect["left"]
    y_min = scaled_crop_rect["top"]
    x_max = x_min + scaled_crop_rect["width"]
    y_max = y_min + scaled_crop_rect["height"]

    # Convert to YOLO format
    x_center = (x_min + x_max) / 2 / orig_width
    y_center = (y_min + y_max) / 2 / orig_height
    width_norm = scaled_crop_rect["width"] / orig_width
    height_norm = scaled_crop_rect["height"] / orig_height

    # Create output directory for gemini labels inside ROOT
    gemini_dir = os.path.join(settings.ROOT, "gemini-labels")
    os.makedirs(gemini_dir, exist_ok=True)

    # Convert RGBA to RGB if necessary
    if original_img.mode in ("RGBA", "LA"):
        original_img = original_img.convert("RGB")

    # Append to existing label file
    label_path = os.path.join(gemini_dir, f"{base_filename}.txt")
    with open(label_path, "a") as f:
        f.write(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n"
        )

    # Save original image only if it doesn't exist
    image_path = os.path.join(gemini_dir, f"{base_filename}.jpg")
    if not os.path.exists(image_path):
        original_img.save(image_path)
        st.success(f"Saved original image: {image_path}")
    else:
        st.info(f"Original image already exists: {image_path}")

    # Count total annotations
    with open(label_path, "r") as f:
        num_annotations = len(f.readlines())

    st.success(
        f"Appended Gemini prediction (total: {num_annotations}):\n- {label_path}")
    st.info(f"Class: {class_name} (ID: {class_id})\n"
            f"YOLO: [{class_id}, {x_center:.4f}, {y_center:.4f}, {width_norm:.4f}, {height_norm:.4f}]")


# Main Gemini Analysis Section
if st.session_state.cropped_img and st.session_state.crop_rect:
    st.image(st.session_state.cropped_img,
             caption="Cropped Image for Gemini Analysis", width=365)

    if st.button("Analyze with Gemini") or st.session_state.analysis_done:
        if not st.session_state.analysis_done:
            with st.spinner("Gemini is analyzing the dental image..."):
                try:
                    # Convert cropped image to bytes
                    image_bytes = pil_to_bytes(st.session_state.cropped_img)

                    # IMPROVED: Single pass with better prompt (no autoregression)
                    st.info("Analyzing dental image...")

                    response = gemini_model.generate_content(
                        [IMPROVED_PROMPT, {
                            "mime_type": "image/jpeg", "data": image_bytes}],
                        generation_config=generation_config
                    )

                    # Safe text extraction
                    final_class = ""
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and candidate.content.parts:
                            final_class = candidate.content.parts[0].text.strip(
                            )
                        elif hasattr(response, 'text'):
                            final_class = response.text.strip()

                    if not final_class:
                        st.warning(
                            f"Analysis failed. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
                        raise Exception(
                            "Could not get valid response from Gemini")

                    # Clean up the response - remove quotes, markdown, extra text
                    final_class = final_class.replace("'", "").replace(
                        '"', "").replace("`", "").strip()

                    # Extract only the class name if there's additional text
                    for dental_class in DENTAL_CLASSES:
                        if dental_class.lower() in final_class.lower():
                            final_class = dental_class
                            break

                    st.success(f"Classification: {final_class}")

                    st.session_state.gemini_label = final_class
                    st.session_state.analysis_done = True

                except Exception as e:
                    st.error(f"Error during Gemini analysis: {e}")
                    if 'response' in locals():
                        st.error(f"Response details: {response}")
                        # Show the raw response for debugging
                        if response.candidates:
                            st.code(str(response.candidates[0]))
                    st.info(
                        "Try re-cropping the image or uploading a clearer dental image.")

        # Display results
        if st.session_state.gemini_label:
            class_id = map_class_to_id(st.session_state.gemini_label)

            st.success("Gemini Classification Complete!")
            st.markdown(f"### Detected: **{st.session_state.gemini_label}**")
            st.markdown(f"**Class ID:** {class_id} (out of 44 classes)")

            # Highlight the detected class
            with st.expander("View all 44 classes"):
                for idx, cls in enumerate(DENTAL_CLASSES):
                    if cls == st.session_state.gemini_label:
                        st.markdown(f"**{idx}: {cls}** <- Detected")
                    else:
                        st.markdown(f"{idx}: {cls}")

            # Save button
            if st.button("Save Gemini Label + Original Image"):
                if st.session_state.original_uploaded_img is not None:
                    save_gemini_yolo_label(
                        st.session_state.img_file,
                        st.session_state.crop_rect,
                        st.session_state.gemini_label,
                        st.session_state.original_uploaded_img
                    )
                else:
                    st.error(
                        "Original image not found. Please upload an image first.")

else:
    st.warning(
        "Please crop an area from the detected image first to analyze with Gemini.")

# GEMINI DETECTION SECTION
st.title("Gemini AI Detection - Full Image Analysis")

# Comprehensive detection prompt with structured output requirements
GEMINI_DETECTION_PROMPT = """You are an expert dental diagnostician with computer vision capabilities. Analyze this dental image and detect ALL visible dental features with MAXIMUM PRECISION.

CLASSES (0-43):
TEETH: 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48
CONDITIONS: amalgam filling, calculus, fixed prosthesis, incisive papilla, non-carious lesion, palatine raphe, staining or visible changes without cavitation, temporary restoration, tongue, tooth coloured filling, visible changes with cavitation, visible changes with microcavitation

DETECTION INSTRUCTIONS:
1. Identify each visible tooth and dental condition
2. For EACH feature, draw a tight bounding box around it
3. Measure the box position as percentages of image dimensions (0-100%)
4. Report coordinates with HIGH PRECISION (use decimals: 45.7%, not 45%)

BOUNDING BOX FORMAT: [x_min, y_min, x_max, y_max]
- x_min: left edge position (%)
- y_min: top edge position (%)
- x_max: right edge position (%)
- y_max: bottom edge position (%)
- USE DECIMAL VALUES for accuracy (e.g., 23.5, 47.8, 65.2)

CONFIDENCE LEVELS:
- high: clearly visible, no ambiguity
- medium: visible but some uncertainty
- low: barely visible or unclear

Return ONLY valid JSON array with precise measurements:
[
  {"class": "tooth_number_or_condition", "bbox": [x_min, y_min, x_max, y_max], "confidence": "high/medium/low"}
]

EXAMPLE OUTPUT:
[
  {"class": "48", "bbox": [15.3, 22.7, 28.9, 51.2], "confidence": "high"},
  {"class": "calculus", "bbox": [31.5, 48.2, 39.7, 56.8], "confidence": "medium"}
]

Prioritize visible pathologies, then teeth. Maximum 15 most significant detections."""


def parse_gemini_json(response_text: str):
    """
    Parse JSON response from Gemini with automatic error recovery.
    Attempts to fix common JSON formatting issues and validates detection structure.
    """
    import json
    import re
    
    try:
        # Clean up markdown code blocks and whitespace
        text = response_text.strip()
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        text = text.strip()
        
        # Attempt to auto-close incomplete JSON structures
        if not text.endswith(']'):
            open_brackets = text.count('[') - text.count(']')
            open_braces = text.count('{') - text.count('}')
            
            if open_braces > 0:
                text = text.rstrip(',\n ')
                text += '}' * open_braces
            if open_brackets > 0:
                text += ']' * open_brackets
        
        detections = json.loads(text)
        
        if not isinstance(detections, list):
            return []
        
        # Validate each detection has required fields
        valid = []
        has_integer_coords = False
        
        for det in detections:
            if (isinstance(det, dict) and 
                "class" in det and 
                "bbox" in det and 
                isinstance(det["bbox"], list) and 
                len(det["bbox"]) == 4):
                
                # Check if all coordinates are integers (lack precision)
                bbox = det["bbox"]
                if all(isinstance(coord, (int, float)) and coord == int(coord) for coord in bbox):
                    has_integer_coords = True
                
                valid.append(det)
        
        # Warn user if coordinates lack precision
        if has_integer_coords:
            st.warning("Gemini returned integer coordinates. Bounding boxes may not be precise. Consider re-running or refining the image.")
        
        return valid
        
    except Exception as e:
        st.error(f"JSON parsing error: {e}")
        with st.expander("View raw response"):
            st.code(response_text)
        return []


def create_segmentation_mask(bbox, img_width, img_height):
    """
    Convert bounding box to polygon mask for YOLOv8 segmentation format.
    Returns normalized polygon coordinates as a simple rectangle.
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Convert percentage coordinates to normalized 0-1 range
    x_min_norm = x_min / 100.0
    y_min_norm = y_min / 100.0
    x_max_norm = x_max / 100.0
    y_max_norm = y_max / 100.0
    
    # Create rectangular polygon with 4 corner points
    polygon = [
        x_min_norm, y_min_norm,  # top-left
        x_max_norm, y_min_norm,  # top-right
        x_max_norm, y_max_norm,  # bottom-right
        x_min_norm, y_max_norm   # bottom-left
    ]
    
    return polygon


def draw_detections_with_masks(image, detections, dpi=640):
    """
    Render bounding boxes, semi-transparent masks, and bold labels on the image.
    Uses color-coded overlays for different detections.
    High DPI setting ensures crisp, publication-quality output.
    """
    from PIL import ImageDraw, ImageFont
    import numpy as np
    
    img_width, img_height = image.size
    img_array = np.array(image)
    
    # Create transparent overlay layer for masks
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_main = ImageDraw.Draw(image)
    
    # Distinct color palette for visual separation
    colors = [
        (255, 107, 107), (78, 205, 196), (69, 183, 209), (255, 160, 122),
        (152, 216, 200), (247, 220, 111), (187, 143, 206), (133, 193, 226),
        (248, 183, 57), (82, 183, 136), (255, 99, 132), (54, 162, 235)
    ]
    
    # Load fonts with fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    for idx, det in enumerate(detections):
        class_name = det.get("class", "unknown")
        bbox = det.get("bbox", [])
        confidence = det.get("confidence", "unknown")
        
        if len(bbox) != 4:
            continue
        
        # Convert percentage coordinates to pixel coordinates
        x_min = int(bbox[0] * img_width / 100)
        y_min = int(bbox[1] * img_height / 100)
        x_max = int(bbox[2] * img_width / 100)
        y_max = int(bbox[3] * img_height / 100)
        
        color = colors[idx % len(colors)]
        color_rgba = (*color, 80)
        
        # Draw semi-transparent filled mask
        draw_overlay.rectangle([x_min, y_min, x_max, y_max], fill=color_rgba)
        
        # Draw solid bounding box outline with increased width
        draw_main.rectangle([x_min, y_min, x_max, y_max], outline=color, width=4)
        
        # Prepare label text
        label_text = f"{class_name}"
        conf_text = f"{confidence}"
        
        # Calculate label background dimensions
        bbox_label = draw_main.textbbox((x_min, y_min - 45), label_text, font=font)
        bbox_conf = draw_main.textbbox((x_min, y_min - 22), conf_text, font=font_small)
        
        # Draw solid background for label
        draw_main.rectangle([bbox_label[0]-4, bbox_label[1]-4, bbox_label[2]+4, bbox_label[3]+4], fill=color)
        
        # Draw text with increased weight for visibility
        draw_main.text((x_min, y_min - 45), label_text, fill="white", font=font, stroke_width=2, stroke_fill="black")
        draw_main.text((x_min, y_min - 22), conf_text, fill="white", font=font_small, stroke_width=1, stroke_fill="black")
    
    # Composite overlay onto main image
    image = image.convert('RGBA')
    image = Image.alpha_composite(image, overlay)
    image = image.convert('RGB')
    
    return image


def save_yolo_segmentation_format(detections, img_width, img_height, base_filename):
    """
    Export detections to YOLOv8 segmentation format text file.
    Format: <class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
    """
    output_dir = os.path.join(settings.ROOT, "gemini-detection-labels")
    os.makedirs(output_dir, exist_ok=True)
    
    label_path = os.path.join(output_dir, f"{base_filename}.txt")
    
    with open(label_path, "w") as f:
        for det in detections:
            class_name = det.get("class", "")
            bbox = det.get("bbox", [])
            
            if len(bbox) != 4:
                continue
            
            # Map class name to numeric ID
            class_id = map_class_to_id(class_name)
            if class_id == -1:
                st.warning(f"Unknown class: {class_name}")
                continue
            
            # Generate polygon mask from bounding box
            polygon = create_segmentation_mask(bbox, img_width, img_height)
            
            # Write in YOLOv8 segmentation format
            line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in polygon])
            f.write(line + "\n")
    
    return label_path


# Main Gemini Detection Interface
if st.session_state.original_uploaded_img is not None:
    st.markdown("---")
    st.info("Gemini will analyze the complete image and detect all dental features with bounding boxes and segmentation masks.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(st.session_state.original_uploaded_img, use_container_width=True)
    
    if st.button("Run Gemini Detection", type="primary", use_container_width=True):
        with st.spinner("Gemini is analyzing the dental image..."):
            try:
                # Convert image to bytes for API transmission
                image_bytes = pil_to_bytes(st.session_state.original_uploaded_img)
                
                # Configure Gemini generation parameters for balanced performance
                # Lower temperature for more focused, deterministic outputs
                # Top-p and top-k control token sampling diversity
                detection_config = genai.types.GenerationConfig(
                    temperature=0.2,      # Low temperature for consistent, focused responses
                    top_p=0.95,          # Nucleus sampling - considers top 95% probability mass
                    top_k=40,            # Limits sampling to top 40 most likely tokens
                    max_output_tokens=8192,
                )
                
                # Send request to Gemini API
                response = gemini_model.generate_content(
                    [GEMINI_DETECTION_PROMPT, {"mime_type": "image/jpeg", "data": image_bytes}],
                    generation_config=detection_config
                )
                
                # Extract text response from API result
                response_text = ""
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content.parts:
                        response_text = candidate.content.parts[0].text.strip()
                    elif hasattr(response, 'text'):
                        response_text = response.text.strip()
                
                if not response_text:
                    st.error("No response from Gemini")
                    st.stop()
                
                # Parse JSON response into detection objects
                detections = parse_gemini_json(response_text)
                
                if not detections:
                    st.warning("No valid detections found")
                    st.stop()
                
                st.success(f"Detected {len(detections)} dental features!")
                
                # Display detailed detection information
                with st.expander(f"View All {len(detections)} Detections"):
                    for i, det in enumerate(detections, 1):
                        class_name = det.get('class', 'unknown')
                        class_id = map_class_to_id(class_name)
                        confidence = det.get('confidence', 'N/A')
                        bbox = det.get('bbox', [])
                        
                        st.markdown(f"""
                        **{i}. {class_name}** (ID: {class_id})
                        - Confidence: {confidence}
                        - BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]
                        """)
                
                # Get image dimensions for coordinate conversion
                img_width, img_height = st.session_state.original_uploaded_img.size
                
                # Save labels in YOLOv8 format
                base_filename = st.session_state.original_image_filename or get_unique_filename()
                label_path = save_yolo_segmentation_format(
                    detections, img_width, img_height, base_filename
                )
                st.success(f"Labels saved: `{label_path}`")
                
                # Render detections on image copy with high DPI
                detected_image = draw_detections_with_masks(
                    st.session_state.original_uploaded_img.copy(),
                    detections,
                    dpi=640
                )
                
                # Create output directory for saving both original and detected images
                output_img_dir = os.path.join(settings.ROOT, "gemini-detected-images")
                os.makedirs(output_img_dir, exist_ok=True)
                
                # Save original image in the same folder
                original_img_path = os.path.join(output_img_dir, f"{base_filename}_original.jpg")
                original_img_to_save = st.session_state.original_uploaded_img.copy()
                if original_img_to_save.mode in ("RGBA", "LA"):
                    original_img_to_save = original_img_to_save.convert("RGB")
                original_img_to_save.save(original_img_path, dpi=(640, 640))
                st.success(f"Original image saved: `{original_img_path}`")
                
                # Save annotated image with high DPI
                detected_img_path = os.path.join(output_img_dir, f"{base_filename}_detected.jpg")
                
                if detected_image.mode in ("RGBA", "LA"):
                    detected_image = detected_image.convert("RGB")
                
                detected_image.save(detected_img_path, dpi=(640, 640))
                st.success(f"Detected image saved: `{detected_img_path}`")
                
                # Display annotated result
                with col2:
                    st.subheader("Detected Image")
                    st.image(detected_image, use_container_width=True)
                
                # Show detection statistics
                st.markdown("---")
                st.subheader("Detection Summary")
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Total Detections", len(detections))
                
                with col_b:
                    high_conf = sum(1 for d in detections if d.get('confidence') == 'high')
                    st.metric("High Confidence", high_conf)
                
                with col_c:
                    unique_classes = len(set(d.get('class') for d in detections))
                    st.metric("Unique Classes", unique_classes)
                
            except Exception as e:
                st.error(f"Error during detection: {e}")
                import traceback
                with st.expander("View error details"):
                    st.code(traceback.format_exc())

else:
    st.warning("Please upload an image first to use Gemini Detection.")