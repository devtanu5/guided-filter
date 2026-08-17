import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

from core.filter import GuidedFilter
from core.applications import detail_enhancement, dehaze, joint_filtering
from cv.image import to_32F, to_8U

# Setup page layout and title
st.set_page_config(
    page_title="Guided Filter Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(135deg, #FF3366, #FF9933, #33CCFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        color: #8888aa;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #FF3366, #FF9933);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border-radius: 30px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 51, 102, 0.4);
    }
</style>
""", unsafe_allowed_html=True)

st.markdown('<h1 class="main-title">✨ Guided Image Filter Studio</h1>', unsafe_allowed_html=True)
st.markdown('<p class="sub-title">An interactive playground for Edge-Aware Image Smoothing, Detail Enhancement, and Single-Image Dehazing based on He, Sun & Tang (TPAMI \'12)</p>', unsafe_allowed_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("## 🛠️ Configuration")

# Task Selection
task = st.sidebar.selectbox(
    "Select Application / Task",
    ["Edge-Preserving Smoothing", "Detail Enhancement", "Single-Image Dehazing", "Joint Guided Filtering"]
)

# Upload / Preset selector
st.sidebar.markdown("---")
st.sidebar.markdown("### 🖼️ Image Source")
source_type = st.sidebar.radio("Select Image Source", ["Presets", "Upload custom image"])

uploaded_file = None
preset_name = None

if source_type == "Presets":
    preset_name = st.sidebar.selectbox("Select Preset Image", ["Cat (Gray)", "Lenna (Color)"])
    if preset_name == "Cat (Gray)":
        image_path = "data/cat.png"
    else:
        image_path = "data/Lenna.png"
    
    # Load preset
    img_bgr = cv2.imread(image_path)
    if img_bgr is not None:
        if len(img_bgr.shape) == 3 and preset_name == "Lenna (Color)":
            img_input = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_input = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        st.sidebar.error("Error: Preset image not found. Please verify project data files.")
        img_input = None
else:
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Check if color or gray is preferred
        color_mode = st.sidebar.checkbox("Load as Color Image (RGB)", value=True)
        if color_mode:
            img_input = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_input = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_input = None

# If joint guided filtering, load a guide image
guide_input = None
if task == "Joint Guided Filtering" and img_input is not None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Guide Image")
    guide_source = st.sidebar.radio("Select Guide Source", ["Use Input Image itself (self-guided)", "Upload custom guide"])
    if guide_source == "Use Input Image itself (self-guided)":
        guide_input = img_input
    else:
        guide_file = st.sidebar.file_uploader("Upload Guide Image", type=["png", "jpg", "jpeg"])
        if guide_file is not None:
            guide_bytes = np.asarray(bytearray(guide_file.read()), dtype=np.uint8)
            guide_bgr = cv2.imdecode(guide_bytes, cv2.IMREAD_COLOR)
            if len(img_input.shape) == 3:
                guide_input = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2RGB)
            else:
                guide_input = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY)

# Task-specific Parameters
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Filter Parameters")

if task == "Edge-Preserving Smoothing":
    radius = st.sidebar.slider("Radius (r)", min_value=1, max_value=30, value=8, step=1)
    eps = st.sidebar.slider("Epsilon (ε)", min_value=0.0001, max_value=1.0, value=0.04, step=0.0001, format="%.4f")
    
elif task == "Detail Enhancement":
    radius = st.sidebar.slider("Radius (r)", min_value=1, max_value=30, value=8, step=1)
    eps = st.sidebar.slider("Epsilon (ε)", min_value=0.0001, max_value=0.5, value=0.04, step=0.0001, format="%.4f")
    factor = st.sidebar.slider("Amplification Factor", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
    color_preserve = st.sidebar.checkbox("Color-Space Preservation (Luminance Only)", value=True)

elif task == "Single-Image Dehazing":
    dcp_radius = st.sidebar.slider("DCP Patch Radius", min_value=1, max_value=25, value=7, step=1)
    omega = st.sidebar.slider("Haze Preservation Factor (ω)", min_value=0.50, max_value=1.0, value=0.95, step=0.01)
    t0 = st.sidebar.slider("Transmission Lower Bound (t0)", min_value=0.05, max_value=0.5, value=0.10, step=0.01)
    guided_radius = st.sidebar.slider("Guided Refinement Radius", min_value=5, max_value=100, value=40, step=5)
    guided_eps = st.sidebar.slider("Guided Refinement Epsilon (ε)", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-5, format="%.5f")

elif task == "Joint Guided Filtering":
    radius = st.sidebar.slider("Radius (r)", min_value=1, max_value=30, value=4, step=1)
    eps = st.sidebar.slider("Epsilon (ε)", min_value=0.0001, max_value=0.5, value=0.01, step=0.0001, format="%.4f")


# ----------------- MAIN WORKSPACE -----------------
if img_input is None:
    st.info("👋 Welcome! Please select a preset image or upload your own to begin.")
else:
    # Perform Filtering
    with st.spinner("Processing image..."):
        if task == "Edge-Preserving Smoothing":
            gf = GuidedFilter(img_input, radius=radius, eps=eps)
            out_rgb = gf.filter(img_input)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📤 Original Image")
                st.image(img_input, use_container_width=True)
            with col2:
                st.markdown("### ✨ Smoothed Image")
                st.image(np.clip(out_rgb, 0.0, 1.0), use_container_width=True)
                
        elif task == "Detail Enhancement":
            out_rgb = detail_enhancement(
                img_input,
                radius=radius,
                eps=eps,
                factor=factor,
                color_space_preserve=color_preserve
            )
            
            # Extract detail layer for visualization
            if len(img_input.shape) == 3 and color_preserve:
                img_8u = to_8U(to_32F(img_input))
                ycrcb = cv2.cvtColor(img_8u, cv2.COLOR_RGB2YCrCb)
                y = ycrcb[:, :, 0].astype(np.float32) / 255.0
                gf = GuidedFilter(y, radius=radius, eps=eps)
                base_y = gf.filter(y)
                detail = y - base_y
            else:
                gf = GuidedFilter(img_input, radius=radius, eps=eps)
                base = gf.filter(img_input)
                detail = to_32F(img_input) - base
                if len(detail.shape) == 3:
                    detail = np.mean(detail, axis=2) # Average color channels for detail visualization
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📤 Original Image")
                st.image(img_input, use_container_width=True)
            with col2:
                st.markdown("### ✨ Enhanced Image")
                st.image(np.clip(out_rgb, 0.0, 1.0), use_container_width=True)
            
            # Show details layer
            st.markdown("### 🔍 Extracted Details Map")
            # Map detail values (around 0) to [0, 255] for visual scanning, with 128 as grey baseline
            detail_viz = np.clip((detail * factor + 0.5), 0.0, 1.0)
            st.image(detail_viz, caption=f"Visualizing detail layer multiplied by factor={factor}", use_container_width=True)
            
        elif task == "Single-Image Dehazing":
            # Dehaze returns dehazed image
            # Let's also extract transmission maps for step-by-step visualization
            I = to_32F(img_input)
            dark = estimate_dark_channel(I, radius=dcp_radius)
            A = estimate_atmospheric_light(I, dark)
            t_rough = estimate_transmission(I, A, radius=dcp_radius, omega=omega)
            
            if len(I.shape) == 3:
                guide = cv2.cvtColor(to_8U(I), cv2.COLOR_RGB2GRAY)
                guide = to_32F(guide)
            else:
                guide = I
                
            gf = GuidedFilter(guide, radius=guided_radius, eps=guided_eps)
            t_refined = gf.filter(t_rough)
            t_refined = np.clip(t_refined, t0, 1.0)
            
            out_rgb = dehaze(
                img_input,
                radius=dcp_radius,
                eps=guided_eps,
                omega=omega,
                t0=t0,
                guided_radius=guided_radius,
                guided_eps=guided_eps
            )
            
            tab1, tab2 = st.tabs(["🖼️ Main Dehazing Result", "🔍 DCP Intermediate Outputs"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🌫️ Original Hazy Image")
                    st.image(img_input, use_container_width=True)
                with col2:
                    st.markdown("### ✨ Dehazed Scene Radiance")
                    st.image(out_rgb, use_container_width=True)
                    
            with tab2:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("#### 1. Dark Channel Prior")
                    st.image(dark, caption="DCP Map", use_container_width=True, clamp=True)
                with col2:
                    st.markdown("#### 2. Rough Transmission Map")
                    st.image(t_rough, caption="Coarse Transmission Map", use_container_width=True, clamp=True)
                with col3:
                    st.markdown("#### 3. Refined Transmission Map")
                    st.image(t_refined, caption="Refined with Guided Filter", use_container_width=True, clamp=True)
                    
                st.info(f"Estimated Atmospheric Light A: RGB({int(A[0]*255)}, {int(A[1]*255)}, {int(A[2]*255)})")
                
        elif task == "Joint Guided Filtering":
            if guide_input is None:
                st.warning("⚠️ Please provide/upload a guide image in the sidebar.")
                out_rgb = None
            else:
                out_rgb = joint_filtering(img_input, guide_input, radius=radius, eps=eps)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📤 Input / Noisy Target")
                    st.image(img_input, use_container_width=True)
                with col2:
                    st.markdown("### 🧭 Guide Image")
                    st.image(guide_input, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### ✨ Joint Filtered Output")
                st.image(np.clip(out_rgb, 0.0, 1.0), use_container_width=True)

        # ----------------- DOWNLOAD RESULTS -----------------
        if out_rgb is not None:
            st.markdown("---")
            st.markdown("### 💾 Export Result")
            # Convert float32 RGB to 8U BGR, then encode as PNG
            out_img = Image.fromarray(to_8U(np.clip(out_rgb, 0.0, 1.0)))
            buf = io.BytesIO()
            out_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Processed Image",
                data=byte_im,
                file_name=f"guided_filter_{task.lower().replace(' ', '_')}.png",
                mime="image/png"
            )
