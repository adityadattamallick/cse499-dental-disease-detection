"""
Dental Image Detection and Annotation Application
=================================================
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

# ==================== PAGE CONFIGURATION ====================
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
st.title("Dental Image Detection and Prediction Using YOLOv8 and SAM")

# ==================== SESSION STATE INITIALIZATION ====================
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

# ==================== SIDEBAR: MODEL CONFIGURATION ====================
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

# ==================== SIDEBAR: IMAGE SOURCE ====================
st.sidebar.header("Image Config")
source_radio = st.sidebar.radio("Select Your Source", settings.SOURCES_LIST)

# ==================== IMAGE UPLOAD AND DETECTION ====================
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

# ==================== ANNOTATION TOOL ====================
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
st.header("Predicted Image (Original Size):")
st.image(img_file, width=img_width)

# Interactive cropping tool
st.header("Edit Bounding Boxes:")
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

# Label input
st.header("Enter Label for the bounding box:")
st.link_button(
    "Label Class Details As Per The Dataset - Enter Any Number Between 0 to 43 As the Label that Corresponds to Your Correction in the Given Table",
    "https://github.com/adityadattamallick/cse499-dental-disease-detection?tab=readme-ov-file#dataset-details",
)
st.session_state.label = st.text_input(
    "Enter Label here:", st.session_state.get("label", "")
)
st.header(f"Entered Label: '{st.session_state.label}'")

# ==================== UTILITY FUNCTIONS ====================


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


# ==================== SAVE BUTTONS SECTION ====================

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

# ==================== GEMINI AI INTEGRATION ====================
st.title("Gemini 2.5 Flash - Cropped Tooth Disease Detection")

# Load Gemini API key from environment
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in .env")
    st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Get latest cropped image for analysis
latest_image_path = get_latest_file()
if latest_image_path:
    st.info(f"Latest cropped image: `{latest_image_path}`")
    img = Image.open(latest_image_path)
    st.image(img, caption="Latest Cropped Image", use_container_width=True)

    # Convert image to bytes once for reuse in all Gemini calls
    image_bytes = pil_to_bytes(img)

    # Gemini disease detection
    if st.button("Generate Gemini Label") or st.session_state.analysis_done:
        if not st.session_state.analysis_done:
            st.info("Analyzing the cropped image with Gemini...")
            try:
                prompt = "Identify the dental disease in this image and return a single safe YOLO class name."
                resp = gemini_model.generate_content(
                    [prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
                )

                # Extract response text
                raw_label = getattr(resp, "text", "")
                if not raw_label:
                    try:
                        raw_label = resp.generations[0].content
                    except:
                        raw_label = "unknown"

                # Sanitize for YOLO compatibility
                st.session_state.gemini_label = sanitize_class_name(raw_label)
                st.session_state.analysis_done = True
            except Exception as e:
                st.error(f"Error calling Gemini: {e}")

        # Display results and save option
        if st.session_state.gemini_label:
            st.success("Gemini Detected Disease:")
            st.markdown(f"**{st.session_state.gemini_label}**")

            if st.button("Save Gemini Label and Image"):
                save_gemini_label_and_image(
                    latest_image_path, st.session_state.gemini_label
                )

            # ==================== GEMINI CHAT INTERFACE ====================
            st.divider()
            st.subheader("Chat with Gemini about this disease")

            # Provide default question about detected disease
            default_prompt = f"Explain more about the dental disease '{st.session_state.gemini_label}' in simple terms."
            user_input = st.text_input(
                "Ask a question about the disease:", value=default_prompt
            )

            if user_input:
                try:
                    # Send question with image context to Gemini
                    resp = gemini_model.generate_content(
                        [user_input, {"mime_type": "image/jpeg", "data": image_bytes}]
                    )

                    # Extract response
                    reply = getattr(resp, "text", "")
                    if not reply:
                        try:
                            reply = resp.generations[0].content
                        except:
                            reply = "No response."

                    # Add to chat history
                    st.session_state.chat_history.append((user_input, reply.strip()))
                except Exception as e:
                    st.error(f"Error in Gemini chat: {e}")

            # Display chat history
            for q, a in st.session_state.chat_history:
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Gemini:** {a}")
else:
    st.warning("No images found in the `custom-labels/` folder.")
