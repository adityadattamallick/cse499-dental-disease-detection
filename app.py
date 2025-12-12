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

# SIDEBAR: MODEL CONFIGURATION
st.sidebar.header("DL Model Configuration")

# Model type selection
model_type = st.sidebar.radio("Choose Task", ["Detection", "Segmentation"])

# Confidence threshold slider
confidence = float(st.sidebar.slider("Confidence Threshold (%)", 25, 100, 30)) / 100

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
st.sidebar.header("Image Config")
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
                    default_image, caption="Default Image", use_container_width=True
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
                st.image(source_img, caption="Uploaded Image Successfully")
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
                use_container_width=True,
            )
        else:
            # Run detection when button is clicked
            if st.sidebar.button("Detect Disease and Stuff"):
                res = model.predict(uploaded_image, conf=confidence)
                res_plotted = res[0].plot(labels=True)[:, :, ::-1]
                st.image(
                    res_plotted, caption="Detected Image", use_container_width=True
                )
                # Save detected image for annotation
                st.session_state.img_file = Image.fromarray(res_plotted)
                st.session_state.img_file.save("detected_image.jpg")

# ANNOTATION TOOL 
st.title("Annotation Tool")

# Cropper configuration
realtime_update = st.checkbox("Update in Real Time", value=True)
box_color = st.color_picker("Box Color", value="#0b4cd9")
aspect_choice = st.radio("Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
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
st.image(img_file, width=img_width)

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
    st.session_state.crop_rect = crop_result
    # Manually crop using rectangle coordinates from original image
    left = crop_result["left"]
    top = crop_result["top"]
    width = crop_result["width"]
    height = crop_result["height"]
    st.session_state.cropped_img = img_file.crop(
        (left, top, left + width, top + height)
    )

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
st.header(f"Selected Label: {st.session_state.label} - {selected_label.split(': ', 1)[1]}")

# UTILITY FUNCTIONS


def get_unique_filename():
    """Generate a unique filename using UUID."""
    return str(uuid.uuid4())


def save_img_label_yolo_format(img_width, img_height, crop_rect, label, cropped_img):
    """
    Save cropped image and corresponding YOLO format label.

    YOLO format: <class> <x_center> <y_center> <width> <height>
    All coordinates are normalized (0-1) relative to original image dimensions.

    Args:
        img_width (int): Original image width in pixels
        img_height (int): Original image height in pixels
        crop_rect (dict): Dictionary with 'left', 'top', 'width', 'height' from st_cropper
        label (str): Class label for the bounding box
        cropped_img (PIL.Image): The cropped image to save
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

    # Convert RGBA to RGB if necessary
    if cropped_img.mode in ("RGBA", "LA"):
        cropped_img = cropped_img.convert("RGB")

    # Save YOLO format label file
    label_path = os.path.join("custom-labels", f"{filename}.txt")
    with open(label_path, "w") as f:
        f.write(
            f"{label} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n"
        )

    # Save cropped image
    image_path = os.path.join("custom-labels", f"{filename}.jpg")
    cropped_img.save(image_path)

    st.success(f"Saved crop and label:\n- {image_path}\n- {label_path}")
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
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder)
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


# Save image and label

def save_original_with_yolo_label(
    img_width, img_height, crop_rect, label, original_img
):
    """
    Save the original full-size image along with YOLO format label.
    This preserves the complete context while still marking the detected region.

    Args:
        img_width (int): Original image width in pixels
        img_height (int): Original image height in pixels
        crop_rect (dict): Dictionary with 'left', 'top', 'width', 'height' from st_cropper
        label (str): Class label for the bounding box
        original_img (PIL.Image): The original full-size image to save
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

    # Create output directory for original images
    os.makedirs("original-labels", exist_ok=True)

    # Convert RGBA to RGB if necessary
    if original_img.mode in ("RGBA", "LA"):
        original_img = original_img.convert("RGB")

    # Save YOLO format label file
    label_path = os.path.join("original-labels", f"{filename}.txt")
    with open(label_path, "w") as f:
        f.write(
            f"{label} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n"
        )

    # Save original full-size image
    image_path = os.path.join("original-labels", f"{filename}.jpg")
    original_img.save(image_path)

    st.success(f"Saved original image and label:\n- {image_path}\n- {label_path}")
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
    Save YOLO format label with Gemini-predicted class and original full image.
    Scales crop coordinates from detected image to original image dimensions.
    """
    # Map class name to ID
    class_id = map_class_to_id(class_name)
    
    if class_id == -1:
        st.error(f"Could not map '{class_name}' to any of the 44 classes. Using class 0 as fallback.")
        class_id = 0
    
    filename = get_unique_filename()
    
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
    
    # Convert to YOLO format (normalized to ORIGINAL image dimensions)
    x_center = (x_min + x_max) / 2 / orig_width
    y_center = (y_min + y_max) / 2 / orig_height
    width_norm = scaled_crop_rect["width"] / orig_width
    height_norm = scaled_crop_rect["height"] / orig_height
    
    # Create output directory
    os.makedirs("gemini-labels", exist_ok=True)
    
    # Convert RGBA to RGB if necessary
    if original_img.mode in ("RGBA", "LA"):
        original_img = original_img.convert("RGB")
    
    # Save YOLO format label file
    label_path = os.path.join("gemini-labels", f"{filename}.txt")
    with open(label_path, "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
    
    # Save original image
    image_path = os.path.join("gemini-labels", f"{filename}.jpg")
    original_img.save(image_path)
    
    st.success(f"Saved Gemini prediction:\n- Image: {image_path}\n- Label: {label_path}")
    st.info(f"Class: {class_name} (ID: {class_id})\n"
            f"YOLO: [{class_id}, {x_center:.4f}, {y_center:.4f}, {width_norm:.4f}, {height_norm:.4f}]\n"
            f"Original img: {orig_width}x{orig_height}, Detected img: {detected_width}x{detected_height}")


# Main Gemini Analysis Section
if st.session_state.cropped_img and st.session_state.crop_rect:
    st.image(st.session_state.cropped_img, caption="Cropped Image for Gemini Analysis", width=400)
    
    if st.button("Analyze with Gemini") or st.session_state.analysis_done:
        if not st.session_state.analysis_done:
            with st.spinner("Gemini is analyzing the dental image..."):
                try:
                    # Convert cropped image to bytes
                    image_bytes = pil_to_bytes(st.session_state.cropped_img)
                    
                    # IMPROVED: Single pass with better prompt (no autoregression)
                    st.info("Analyzing dental image...")
                    
                    response = gemini_model.generate_content(
                        [IMPROVED_PROMPT, {"mime_type": "image/jpeg", "data": image_bytes}],
                        generation_config=generation_config
                    )
                    
                    # Safe text extraction
                    final_class = ""
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and candidate.content.parts:
                            final_class = candidate.content.parts[0].text.strip()
                        elif hasattr(response, 'text'):
                            final_class = response.text.strip()
                    
                    if not final_class:
                        st.warning(f"Analysis failed. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
                        raise Exception("Could not get valid response from Gemini")
                    
                    # Clean up the response - remove quotes, markdown, extra text
                    final_class = final_class.replace("'", "").replace('"', "").replace("`", "").strip()
                    
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
                    st.info("Try re-cropping the image or uploading a clearer dental image.")
        
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
                    st.error("Original image not found. Please upload an image first.")
            
            # Chat interface section
            st.divider()
            st.subheader("Chat with Gemini about this diagnosis")
            
            default_question = f"Explain '{st.session_state.gemini_label}' in simple terms. What causes it and what treatment is recommended?"
            user_input = st.text_input("Ask a question:", value=default_question)
            
            if st.button("Ask Gemini") and user_input:
                try:
                    image_bytes = pil_to_bytes(st.session_state.cropped_img)
                    resp = gemini_model.generate_content(
                        [user_input, {"mime_type": "image/jpeg", "data": image_bytes}],
                        generation_config=chat_generation_config
                    )
                    reply = getattr(resp, "text", "No response received.")
                    st.session_state.chat_history.append((user_input, reply.strip()))
                except Exception as e:
                    st.error(f"Chat error: {e}")
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("### Chat History")
                for q, a in st.session_state.chat_history:
                    st.markdown(f"**You:** {q}")
                    st.markdown(f"**Gemini:** {a}")
                    st.divider()
else:
    st.warning("Please crop an area from the detected image first to analyze with Gemini.")