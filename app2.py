import streamlit as st
import os
from PIL import Image
from backend2 import DiagnosisPipeline

st.set_page_config(
    page_title="EcoScan",
    page_icon="🔧",
    layout="centered"
)

st.markdown("""
<style>
    /* Main gradient background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Input section */
    .input-container {
        background: linear-gradient(135deg, #1e2a3a 0%, #2c3e50 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Results section */
    .results-header {
        background: linear-gradient(135deg, #0f4c75 0%, #1b262c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        text-align: center;
        font-weight: 600;
        font-size: 1.3rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Info cards */
    [data-testid="stInfo"] {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: #ecf0f1;
    }
    
    /* Recommendation box */
    .recommendation-box {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        color: #ecf0f1;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0f4c75 0%, #3282b8 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        width: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(50, 130, 184, 0.5);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Text area */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.05);
        color: #ecf0f1;
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Warning message */
    .stWarning {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Error message */
    .stError {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Labels */
    .stMarkdown, label {
        color: #ecf0f1;
    }
    
    /* Info icon color */
    [data-testid="stInfo"] > div:first-child {
        color: #3498db;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>EcoScan</h1>
    <p>Advanced device analysis powered by artificial intelligence</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return DiagnosisPipeline()

with st.spinner("Loading AI Models... This may take a minute"):
    pipeline = load_pipeline()
    st.success("System Ready")

st.markdown('<div class="input-container">', unsafe_allow_html=True)

col_upload, col_describe = st.columns([1, 1])

with col_upload:
    st.markdown("**Upload Device Image**")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

with col_describe:
    st.markdown("**Describe Symptoms**")
    user_comment = st.text_area("", height=100, placeholder="e.g., Screen flickers when moved", label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)

if st.button("Analyze Device", type="primary"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
       
        temp_path = "temp_upload.jpg"
        image.save(temp_path)
       
        with st.spinner("Analyzing visuals and symptoms..."):
            try:
                results = pipeline.analyze_case(temp_path, user_comment)
               
                st.markdown('<div class="results-header">Diagnostic Report</div>', unsafe_allow_html=True)
               
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Device Detected:** {results['device_detected']}")
                with col2:
                    st.info(f"**Visual Condition:** {results['visual_condition']}")
               
                st.markdown("### AI Recommendation")
               
                rec_text = results['recommendation']
                st.markdown(f"""
                <div class="recommendation-box">
                    {rec_text.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
               
                with st.expander("View Raw Debug Data"):
                    st.json(results)
                   
            except Exception as e:
                st.error(f"An error occurred: {e}")
           
            if os.path.exists(temp_path):
                os.remove(temp_path)
               
    else:
        st.warning("Please upload an image first")