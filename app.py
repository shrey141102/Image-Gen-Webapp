# from streamlit_autorefresh import st_autorefresh
import streamlit as st
from utils import generate_image_dalle, generate_image_lime, get_image_download_link, generate_image_with_horde_sync
from dotenv import load_dotenv
import os

load_dotenv()

# APIs
openai_key = os.environ.get("OPENAI_API_KEY")
lime_key = os.environ.get("LIME_API_KEY")
ai_horde_key = os.environ.get("HORDE_API_KEY")

# count = st_autorefresh(interval=2000, limit=1000, key="fizzbuzzcounter")

st.title("AI Image Generation Website")

st.sidebar.header("Model Selection")

with st.sidebar:
    model = st.selectbox(
        "Select AI Model",
        ("Dall-E", "Lime", "Anything Diffusion", "AlbedoBase XL (SDXL)", "Deliberate", "ICBINP - I Can't Believe It's Not Photography", "Dreamshaper", "stable_diffusion", "URPM", "Juggernaut XL") ## "aMUSEd", "google-vertexAI",
    )

    api = st.text_input("Enter your API key here",  placeholder="not needed now (testing)")

    st.write("#")

prompt = st.text_input("Enter prompt for image generation", placeholder="Example: A dog and cat fighting by the beach")

if st.button("Generate Image"):
    if prompt:
        try:
            if model == "Dall-E":
                image = generate_image_dalle(prompt, openai_key)
                st.image(image, caption=f"Generated for prompt: '{prompt}'", width=600)

            elif model == "Lime":
                image = generate_image_lime(prompt, lime_key)
                st.image(image, caption=f"Generated for prompt: '{prompt}'", width=600)

            else:
                image, error = generate_image_with_horde_sync(prompt=prompt, model=model, apikey=ai_horde_key)
                if error:
                    st.error(f"Error generating image: {error}")
                else:
                    st.image(image, caption=f"Generated for prompt: '{prompt}'", width=600)

            if 'history' not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append((prompt, image))
        except Exception as e:
            st.error(f"Error generating image: {e}")
    else:
        st.warning("Please enter a prompt.")

if 'history' in st.session_state:
    if len(st.session_state.history) > 1:
        with st.sidebar:
            st.header("History of Generated Images ⬇️")
            for idx, (prompt, image_url) in enumerate(st.session_state.history):
                st.image(image_url, caption=f"Prompt: '{prompt}'", width=200)


if 'history' in st.session_state and st.session_state.history:
    latest_image = st.session_state.history[-1][1]
    # st.image(latest_image, caption=f"Prompt: '{st.session_state.history[-1][0]}'", width=600)
    get_image_download_link(latest_image)

# ------------------------------------------------------------------------------------------------

st.markdown(
    """
    <style>
        button[title^=Exit]+div [data-testid = stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)
