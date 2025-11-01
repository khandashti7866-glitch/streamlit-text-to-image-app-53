import streamlit as st
from PIL import Image
import torch
from io import BytesIO
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Text-to-Image Generator",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 AI Text-to-Image Generator")
st.markdown("Generate amazing images from text prompts using AI! No API key required.")

# --- Sidebar Options ---
st.sidebar.header("Settings")
image_width = st.sidebar.slider("Image Width", min_value=256, max_value=1024, value=512, step=64)
image_height = st.sidebar.slider("Image Height", min_value=256, max_value=1024, value=512, step=64)
num_images = st.sidebar.slider("Number of Images", 1, 4, 1)
style = st.sidebar.selectbox("Art Style", ["Realistic", "Cartoon", "Anime", "Digital Art"])

use_guidance = st.sidebar.checkbox("Use Guidance Image?", value=False)
guidance_image = None
if use_guidance:
    guidance_image = st.sidebar.file_uploader("Upload an Image for Guidance", type=["png", "jpg", "jpeg"])

# --- Text Prompt Input ---
prompt = st.text_area("Enter your image description here", height=100)

# --- Generate Button ---
generate_button = st.button("Generate Image")

# --- Function to load the model (Cached for speed) ---
@st.cache_resource(show_spinner=False)
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    pipe = pipe.to(device)
    return pipe

# --- Function to load img2img model (for guidance image) ---
@st.cache_resource(show_spinner=False)
def load_img2img_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    pipe = pipe.to(device)
    return pipe

# --- Generate Images ---
if generate_button:
    if not prompt.strip():
        st.warning("Please enter a prompt!")
    else:
        with st.spinner("Generating images..."):
            try:
                images = []
                if guidance_image is not None:
                    # Convert uploaded file to PIL Image
                    guidance_image_pil = Image.open(guidance_image).convert("RGB").resize((image_width, image_height))
                    pipe = load_img2img_model()
                    for _ in range(num_images):
                        result = pipe(prompt=prompt, image=guidance_image_pil, strength=0.7, guidance_scale=7.5).images[0]
                        images.append(result)
                else:
                    # Text-to-Image only
                    pipe = load_model()
                    for _ in range(num_images):
                        result = pipe(prompt=prompt, height=image_height, width=image_width, guidance_scale=7.5).images[0]
                        images.append(result)
                
                # Display Images in Columns
                cols = st.columns(len(images))
                for i, img in enumerate(images):
                    cols[i].image(img, use_column_width=True)
                    # Download Button
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    cols[i].download_button(
                        label="Download",
                        data=buffered.getvalue(),
                        file_name=f"generated_image_{i+1}.png",
                        mime="image/png"
                    )
            except Exception as e:
                st.error(f"Error generating image: {e}")
