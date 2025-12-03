import streamlit as st
import os
from PIL import Image
from backend import DiagnosisPipeline # Import your class from the other file

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Device Diagnoser",
    page_icon="🔧",
    layout="centered"
)

# --- 1. LOAD MODEL (CACHED) ---
# @st.cache_resource is CRITICAL. 
# It runs this function once and keeps the 'pipeline' object in memory.
# Without this, your app would reload the 2GB LLM every time you click a button!
@st.cache_resource
def load_pipeline():
    return DiagnosisPipeline()

st.title("🔧 AI Electronics Diagnostics")
st.write("Upload an image of a device and describe the issue.")

# Load the pipeline immediately when app starts
with st.spinner("Loading AI Models... (This may take a minute)"):
    pipeline = load_pipeline()
    st.success("System Ready!")

# --- 2. USER INPUTS ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
user_comment = st.text_area("Describe the symptoms (e.g., 'Screen flickers when moved')", height=100)

# --- 3. PROCESSING ---
if st.button("Analyze Device", type="primary"):
    if uploaded_file is not None:
        
        # Display the image user uploaded
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Save temp file because your pipeline expects a file path
        temp_path = "temp_upload.jpg"
        image.save(temp_path)
        
        with st.spinner("Analyzing Visuals & Symptoms..."):
            try:
                # Run your pipeline
                results = pipeline.analyze_case(temp_path, user_comment)
                
                # --- 4. DISPLAY RESULTS ---
                st.divider()
                st.subheader("📋 Diagnostic Report")
                
                # Layout: 2 Columns for quick stats
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Device Detected:** {results['device_detected']}")
                with col2:
                    st.info(f"**Visual Condition:** {results['visual_condition']}")
                
                # Show Recommendation
                st.markdown("### 🤖 AI Recommendation")
                
                # We parse the text to make it look like a nice card
                rec_text = results['recommendation']
                st.markdown(f"""
                <div style="background-color: #f0f2f6; color: black; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
                    {rec_text.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
                
                # Show JSON for debugging (optional)
                with st.expander("View Raw Debug Data"):
                    st.json(results)
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
            
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    else:
        st.warning("Please upload an image first.")