from pathlib import Path
import PIL
import ultralytics
import streamlit as st
import settings
import helper
from PIL import Image
import os
from streamlit_cropper import st_cropper
import time


import io
from dotenv import load_dotenv
import google.generativeai as genai

import shutil
from glob import glob

import re 

st.set_page_config(
    page_title="Dental Image Segmentation and Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

page_bg_img = f"""
<style>
.stApp, .stSidebar {{
background: #dce3cf;
background: linear-gradient(358deg, rgba(220, 227, 207, 1) 0%, rgba(211, 227, 211, 1) 50%, rgba(151, 199, 194, 1) 100%);
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# st.set_page_config(
#    page_title="Dental Image Segmentation and Detection",
#    layout="wide",
#    initial_sidebar_state="expanded"
# )

st.title("Dental Image Detection and Prediction Using YOLOv8 and SAM")

# Initialize session states
if 'img_file' not in st.session_state:
    st.session_state.img_file = None

if 'cropped_img' not in st.session_state:
    st.session_state.cropped_img = None

if 'label' not in st.session_state:
    st.session_state.label = ""

st.sidebar.header("DL Model Configuration")

model_type = st.sidebar.radio(
    "Choose Task", ['Detection', 'Segmentation']
)

# Default confidence value is 0.30
confidence = float(st.sidebar.slider(
    "Confidence Threshold (In Percentage %)",
    min_value=25,
    max_value=100,
    value=30
)) / 100

if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Error loading model for the following path: {model_path}")
    st.error(ex)

st.sidebar.header("Image Config")

source_radio = st.sidebar.radio(
    "Select Your Source", settings.SOURCES_LIST
)

source_img = None
img_file = None

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image.", type=("jpg", "jpeg", "png", "bmp", "webp")
    )

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                # Placeholder for the progress bar
                progress_bar = st.progress(0)

                if uploaded_image is not None:
                    # Upload progress
                    for percent_complete in range(0, 101, 10):
                        time.sleep(0.1)  # Simulate upload delay
                        progress_bar.progress(percent_complete)
                    progress_bar.empty()

                # Save the uploaded image to session state
                st.session_state.img_file = uploaded_image
                st.image(source_img, caption="Uploaded Image Successfully")

        except Exception as ex:
            st.error(f"Error loading image: {ex}")

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path
            )
            st.image(default_detected_image_path,
                     caption='Detected Image', use_column_width=True)
        else:
            if st.sidebar.button('Detect Disease and Stuff'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot(labels=False)[:, :, ::-1]
                # Update session state with detected image
                st.session_state.img_file = Image.fromarray(res_plotted)
                st.session_state.img_file.save("detected_image.jpg")
                Image.open("detected_image.jpg")

    with st.container():

        # Check `custom-labels` folder exists
        os.makedirs("custom-labels", exist_ok=True)

        # Save bounding coordinates
        def save_img_label_yolo_format(filename, img_width, img_height, bbox, label, cropped_img):
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # Save label to custom-labels folder
            label_path = os.path.join("custom-labels", f"{filename}.txt")
            with open(label_path, "w") as f:
                f.write(
                    f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # Save cropped image to custom-labels folder
            image_path = os.path.join("custom-labels", f"{filename}.jpg")
            cropped_img.save(image_path)

            st.success(
                f"Label and cropped image saved:\n- {label_path}\n- {image_path}")

        # Save cropped image only
        def save_cropped_image(image, filename):
            path = os.path.join("custom-labels", f"{filename}.jpg")
            image.save(path)
            st.success(f"Cropped image saved as {path}")

        st.divider()
        # Streamlit UI setup
        st.title("Annotation Tool")

        # Sidebar controls
        realtime_update = st.checkbox(
            "Update in Real Time", value=True)
        box_color = st.color_picker("Box Color", value='#0b4cd9')
        aspect_choice = st.radio("Aspect Ratio", options=[
            "1:1", "16:9", "4:3", "2:3", "Free"])
        aspect_dict = {"1:1": (1, 1), "16:9": (
            16, 9), "4:3": (4, 3), "2:3": (2, 3), "Free": None}
        aspect_ratio = aspect_dict[aspect_choice]

        if st.session_state.img_file:
            try:
                img = "/Users/adityadatta/Documents/programming/cse499-dental-disease-detection/detected_image.jpg"
                img_file = Image.open(img)
                img_width, img_height = img_file.size

                st.header("Predicted Image:")
                st.image(img_file, width=300)

                st.header("Edit Bounding Boxes:")
                # Cropping using Streamlit cropper tool
                st.session_state.cropped_img = st_cropper(
                    img_file, realtime_update=realtime_update, box_color=box_color, aspect_ratio=aspect_ratio)
                st.header("Cropped Image Preview:")
                st.image(st.session_state.cropped_img, width=300)

                # Input for label
                st.header("Enter Label for the bounding box and then hit enter:")
                st.session_state.label = st.text_input(
                    "Enter Label here: ", st.session_state.label)
                st.header(f"Entered Label:'{st.session_state.label}'")

                # Check if label is empty
                if not st.session_state.label:
                    st.error("Label cannot be empty.")
                else:
                    # Bounding box coordinates
                    left, upper, right, lower = st.session_state.cropped_img.getbbox()
                    bbox = (left, upper, right, lower)

                    # File saving buttons
                    # Use `source_img.name` to get the file name for saving
                    filename = os.path.splitext(source_img.name)[0]
                    if st.button("Save Cropped Image"):
                        save_cropped_image(
                            st.session_state.cropped_img, filename)

                    if st.button("Save YOLO Label"):
                        save_img_label_yolo_format(
                            filename, img_width, img_height, bbox, st.session_state.label, st.session_state.cropped_img)

            except Exception as e:
                st.error(f"An error occurred: {e}")

        st.divider()

        # LLM Feature

        # Setup
        # st.set_page_config(page_title="Gemini Dental Analyzer", layout="wide")
        st.title("Gemini 1.5 Flash - Cropped Tooth Disease Detection")

        # Load the API key
        load_dotenv()
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            st.error("GEMINI_API_KEY not found.")
            st.stop()

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Session State
        if "latest_image_path" not in st.session_state:
            st.session_state.latest_image_path = None
        if "image_saved" not in st.session_state:
            st.session_state.image_saved = False

        # Helpers
        def pil_to_bytes(img: Image.Image) -> bytes:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG")
            return buf.getvalue()

        def sanitize_class_name(name: str) -> str:
            """Ensure YOLO label is a single safe word (underscores allowed)."""
            name = name.strip().lower()
            name = name.replace(" ", "_")          # spaces → underscore
            name = re.sub(r"[^a-z0-9_]", "", name) # remove unwanted chars
            # Keep only first token to enforce single word
            if "_" in name:
                tokens = [t for t in name.split("_") if t]
                name = "_".join(tokens[:2]) if tokens else "unknown"
            return name if name else "unknown"

        def gemini_analyze(image_bytes: bytes, prompt_text: str) -> str:
            image_part = {"mime_type": "image/jpeg", "data": image_bytes}
            resp = model.generate_content([prompt_text, image_part])
            raw_label = getattr(resp, "text", "").strip() or "unknown"
            return sanitize_class_name(raw_label)

        def get_latest_file(folder="custom-labels", ext_list=[".jpg", ".jpeg", ".png"]):
            folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder)
            if not os.path.exists(folder_path):
                return None
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                    if any(f.lower().endswith(ext) for ext in ext_list)]
            if not files:
                return None
            latest_file = max(files, key=os.path.getctime)
            return latest_file

        def save_gemini_label_and_image(latest_image_path, gemini_label):
            # Ensure gemini-label folder exists
            gemini_folder = "gemini-label"
            os.makedirs(gemini_folder, exist_ok=True)

            # Copy cropped image
            image_filename = os.path.basename(latest_image_path)
            dest_image_path = os.path.join(gemini_folder, image_filename)
            shutil.copy2(latest_image_path, dest_image_path)

            # Prepare YOLO label file
            txt_file = os.path.splitext(latest_image_path)[0] + ".txt"
            save_txt_path = os.path.join(gemini_folder, os.path.splitext(image_filename)[0] + ".txt")

            if os.path.exists(txt_file):
                with open(txt_file, "r") as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Replace class with sanitized Gemini label
                        parts[0] = gemini_label
                        new_lines.append(" ".join(parts))
                with open(save_txt_path, "w") as f:
                    f.write("\n".join(new_lines))
                st.success(f"Gemini-corrected label saved:\n{save_txt_path}\nand image copied to:\n{dest_image_path}")
            else:
                # If no original txt exists, create one with dummy bbox
                line = f"{gemini_label} 0.5 0.5 1 1"
                with open(save_txt_path, "w") as f:
                    f.write(line + "\n")
                st.warning(f"No original label found. Created new YOLO label:\n{save_txt_path}")

        # Main
        latest_image_path = get_latest_file()
        if latest_image_path:
            st.info(f"Latest cropped image: `{latest_image_path}`")
            img = Image.open(latest_image_path)
            st.image(img, caption="Latest Cropped Image", use_column_width=True)

            # Initialize session state variables
            if "gemini_label" not in st.session_state:
                st.session_state.gemini_label = None
            if "analysis_done" not in st.session_state:
                st.session_state.analysis_done = False
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Gemini Label Generation
            if st.button("Generate Gemini Label") or st.session_state.analysis_done:
                if not st.session_state.analysis_done:
                    st.info("Analyzing the cropped image with Gemini...")
                    try:
                        prompt = "Identify the dental disease in this image and return a single safe YOLO class name."
                        gemini_label = gemini_analyze(pil_to_bytes(img), prompt)
                        st.session_state.gemini_label = gemini_label
                        st.session_state.analysis_done = True
                    except Exception as e:
                        st.error(f"Error calling Gemini: {e}")

                if st.session_state.gemini_label:
                    st.success("Gemini Detected Disease:")
                    st.markdown(f"**{st.session_state.gemini_label}**")

                    # Save Gemini label and image
                    if st.button("Save Gemini Label and Image"):
                        save_gemini_label_and_image(latest_image_path, st.session_state.gemini_label)

                    # Gemini Chat
                    st.divider()
                    st.subheader("Chat with Gemini about this disease (Assistant to the Dentist/Patient)")

                    # Default prompt based on detected disease
                    default_prompt = f"Explain more about the dental disease '{st.session_state.gemini_label}' in simple terms."

                    user_input = st.text_input("Ask a question about the disease:", value=default_prompt)

                    if user_input:
                        try:
                            image_bytes = pil_to_bytes(img)
                            resp = model.generate_content([user_input, {"mime_type":"image/jpeg","data":image_bytes}])
                            reply = getattr(resp, "text", "").strip() or "No response."
                            st.session_state.chat_history.append((user_input, reply))
                        except Exception as e:
                            st.error(f"Error in Gemini chat: {e}")

                    # Display chat history
                    for q, a in st.session_state.chat_history:
                        st.markdown(f"**You:** {q}")
                        st.markdown(f"**Gemini:** {a}")

        else:
            st.warning("No images found in the `custom-labels/` folder.")