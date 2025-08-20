import os
import io
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
from fpdf import FPDF
import google.generativeai as genai

# Setup
st.set_page_config(page_title="Gemini Flash Multimodal Analyzer", layout="wide")
st.title("Gemini 1.5 Flash Multimodal Dental Disease Analyzer")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Add it to your .env or Streamlit secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")


# Upload images
uploaded_files = st.file_uploader("Upload images", type=["jpg","jpeg","png"], accept_multiple_files=True)
prompt = st.text_input("Enter your prompt", value="Describe the objects in this image.")

# Helpers
def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()

def gemini_analyze(image_bytes: bytes, prompt_text: str) -> str:
    # Gemini Flash accepts image bytes directly with prompt
    image_part = {"mime_type":"image/jpeg", "data": image_bytes}
    resp = model.generate_content([prompt_text, image_part])
    return getattr(resp, "text", "").strip() or "No response."

# Main
if st.button("Analyze Images") and uploaded_files:
    os.makedirs("outputs", exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)

    for idx, uf in enumerate(uploaded_files, start=1):
        pil_img = Image.open(uf)
        st.image(pil_img, caption=f"Image {idx}", use_column_width=True)

        # Gemini call
        st.info("Analyzing with Gemini...")
        try:
            gpt_text = gemini_analyze(pil_to_bytes(pil_img), prompt)
            st.success("Response received from Gemini:")
            st.markdown(gpt_text)
        except Exception as e:
            gpt_text = f"Gemini error: {e}"
            st.error(gpt_text)

        # Add to PDF
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, txt=f"Image {idx}", ln=True)
        temp_img_path = f"outputs/temp_{idx}.jpg"
        pil_img.convert("RGB").save(temp_img_path, "JPEG")
        pdf.image(temp_img_path, w=120)
        pdf.multi_cell(0, 8, txt=f"Prompt:\n{prompt}:\n\nGemini:\n{gpt_text}")
        os.remove(temp_img_path)

    # Save PDF Report
    out_path = "outputs/gemini_report.pdf"
    pdf.output(out_path)
    st.success("PDF Report")
    with open(out_path, "rb") as f:
        st.download_button("Download PDF", f, file_name="gemini_report.pdf")