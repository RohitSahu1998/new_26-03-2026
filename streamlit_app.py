import streamlit as st
import os
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path
import json
import numpy as np

# Import custom modules
from ocr_engine import PaddleOCREngine
from qwen_engine import QwenExtractor
from pipeline_integration import match_qwen_to_ocr, highlight_matches_on_image

# --- Page Config ---
st.set_page_config(page_title="Document Matching App", layout="wide")
st.title("📄 Intelligent Document Extraction")

st.markdown("""
Upload an invoice or document. The app will extract data using Qwen VLM, align it with PaddleOCR coordinates, display a seamlessly highlighted image, and allow you to download the results.
""")

# --- Caching Models ---
@st.cache_resource
def load_models():
    """Load Large AI Models only once across Streamlit reload sessions to save RAM."""
    st.info("Loading AI Models... Please wait, this may take a few moments.")
    ocr = PaddleOCREngine()
    qwen = QwenExtractor()
    return ocr, qwen

try:
    ocr_engine, qwen_engine = load_models()
except Exception as e:
    st.error(f"Failed to load AI models. Please ensure 'ocr_engine.py' and 'qwen_engine.py' paths are correct.\nError Details: {e}")
    st.stop()

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload Document (PDF, PNG, JPG, JPEG)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 1. Save uploaded file to disk so our engines can read it from a path
    temp_path = f"temp_upload_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.success(f"File `{uploaded_file.name}` uploaded successfully!")
    
    if st.button("Process Document and Match Values"):
        with st.spinner("Extracting with PaddleOCR and Qwen. This could take a while..."):
            try:
                # 2. Extract Data
                ocr_results = ocr_engine.extract_text_with_confidence(temp_path)
                qwen_results = qwen_engine.extract_data(temp_path)
                
                # 3. Match Data together
                matched_data = match_qwen_to_ocr(qwen_results, ocr_results)
                
                if not matched_data:
                    st.warning("Could not find any overlapping values between Qwen output and OCR text.")
                else:
                    st.success(f"Perfect! Successfully matched {len(matched_data)} data points from the VLM back to their physical coordinates.")
                    
                    # 4. Highlight and Export Data
                    highlight_image_path = "output_highlighted.jpg"
                    highlight_pdf_path = "output_highlighted.pdf"
                    excel_path = "extracted_data.xlsx"
                    
                    # Generate JPG for the Streamlit UI Preview (The backend will automatically pull Page 1 if multi-page)
                    highlight_matches_on_image(temp_path, matched_data, highlight_image_path)
                    
                    # Generate full Multi-Page PDF specifically for the user to download
                    highlight_matches_on_image(temp_path, matched_data, highlight_pdf_path)
                    
                    # Dump to Excel
                    df = pd.DataFrame(matched_data)
                    df.to_excel(excel_path, index=False)
                    
                    # --- Results UI Layout ---
                    st.divider()
                    st.subheader("Extraction Results")
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("### Highlighted Document")
                        st.image(highlight_image_path, use_container_width=True)
                        
                        # Provide Download Button for Highlighted PDF
                        with open(highlight_pdf_path, "rb") as file:
                            st.download_button(
                                label="✅ Download Highlighted Document (.pdf)",
                                data=file,
                                file_name=f"highlighted_{uploaded_file.name}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            
                    with col2:
                        st.markdown("### Matched Values & Confidence Scores")
                        st.dataframe(df, use_container_width=True)
                        
                        # Provide Download Button for Excel Sheet
                        with open(excel_path, "rb") as file:
                            st.download_button(
                                label="✅ Download Data Dictionary (.xlsx)",
                                data=file,
                                file_name=f"data_{uploaded_file.name}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        
                        # Show raw Qwen extraction for reference
                        with st.expander("Show Raw Qwen Structure"):
                            st.json(qwen_results)

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

    # Optional: Cleanup temp files (can be commented out for debugging)
    # if os.path.exists(temp_path):
    #     os.remove(temp_path)
