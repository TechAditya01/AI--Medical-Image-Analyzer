import streamlit as st
import base64
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure the Google API client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Sample prompt for medical image analysis
sample_prompt = """
You are a medical assistant. Analyze this medical image and provide the following:
1. What condition or issue is shown in the image?
2. What are the key observations?
3. What might be potential treatments or next steps?
4. Are there any concerning features that require immediate attention?
"""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_gpt4_model_for_analysis(filename: str, sample_prompt=sample_prompt):
    # Load the image
    image_data = open(filename, "rb").read()
    
    # Create a model instance with the updated model name
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Generate content
    response = model.generate_content([
        sample_prompt,
        {"mime_type": "image/jpeg", "data": image_data}
    ])
    
    print(response.text)
    return response.text

def chat_eli(query):
    eli5_prompt = "You have to explain the below piece of information to a five years old. \n" + query
    
    # Create a model instance for text-only generation with updated model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Generate content
    response = model.generate_content(eli5_prompt)
    
    return response.text

# Custom CSS for better styling
def local_css():
    st.markdown("""
    <style>
        :root {
            --primary-color: #8a56e8;
            --secondary-color: #f0e6ff;
            --text-color: #333333;
        }
        
        .main {
            background-color: var(--secondary-color);
        }
        
        .stApp {
            max-width: 100%;
            margin: 0 auto;
            padding: 1rem;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #4a2a82; /* Dark purple color */
            padding: 2rem 1rem;
            color: white; /* Text color for better contrast */
        }
        
        /* Make sidebar text white for better visibility */
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] .stMarkdown {
            color: white;
        }
        
        /* Header styling */
        h1 {
            color: var(--primary-color);
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        h3 {
            color: var(--primary-color);
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        /* Button styling */
        .stButton button {
            background-color: var(--primary-color);
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            border: none;
        }
        
        .stButton button:hover {
            background-color: #7040c0;
        }
        
        /* Footer */
        .footer {
            margin-top: 3rem;
            text-align: center;
            color: #6c757d;
            font-size: 0.9rem;
            padding: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
st.set_page_config(
    page_title="HealSmart", 
    page_icon="❤️",
    layout="wide"
)

# Apply custom CSS
local_css()

# Sidebar
with st.sidebar:
    st.markdown("# ❤️ HealSmart")
    st.markdown("---")
    st.markdown("🏠 Home")
    st.markdown("🔍 Medical Image Analysis")

# Main content
st.title("Medical Image Analyzer")
st.markdown("Upload your medical images and get instant AI analysis with simplified explanations.")

# Main upload section
st.header("Upload Medical Image")
st.markdown("Select a medical image (X-ray, MRI, CT scan, etc.) for AI analysis")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.header("Uploaded Image")
    st.image(uploaded_file, caption="", use_container_width=True)
    
    # Save the uploaded file temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Create two columns for buttons
    col1, col2 = st.columns(2)
    
    # Analysis button in first column
    analyze_button = col1.button("🔍 Analyze Image")
    
    # Store analysis results in session state so they persist
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    
    # Analyze the image when button is clicked
    if analyze_button:
        with st.spinner("Analyzing medical image..."):
            analysis = call_gpt4_model_for_analysis("temp_image.jpg")
            st.session_state.analysis_results = analysis
            st.header("Medical Analysis")
            st.markdown(analysis)
    
    # Show the ELI5 button if we have analysis results
    if st.session_state.analysis_results is not None:
        # ELI5 button in second column
        eli5_button = col2.button("👶 Explain Simply")
        if eli5_button:
            with st.spinner("Creating simplified explanation..."):
                simple_explanation = chat_eli(st.session_state.analysis_results)
                st.header("Simplified Explanation")
                st.markdown(simple_explanation)

# Footer
st.markdown("---")
st.markdown("<div class='footer'>HealSmart © 2024 | Powered by Gemini AI</div>", unsafe_allow_html=True)
