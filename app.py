import streamlit as st
import numpy as np
from PIL import Image

from ocr import extract_text
from embedding import create_embeddings
from attention import calculate_attention

st.set_page_config(
    page_title="AI Attention Visualizer",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background-color: #fbfaf7;
}

h1 {
    color: #315c45;
}

h2, h3 {
    color: #315c45;
}

p {
    color: #555555;
}

[data-testid="stFileUploader"] {
    background-color: #f3efe5;
    border: 1px solid #d8d1c2;
    border-radius: 12px;
    padding: 10px;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #f8f5ed;
}

div[data-testid="stProgress"] > div > div > div {
    background-color: #5b9270;
}

.stSuccess {
    background-color: #e8f3eb;
    border-radius: 10px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Attention Visualizer")

st.write(
    "Upload an image to extract text "
    "and visualize attention scores."
)

file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if file:
    image = Image.open(file)

    st.image(image, width=500)

    text = extract_text(image)

    st.subheader("📝 Extracted Text")

    if not text.strip():
        st.error("No text found in the image.")
        st.stop()

    st.write(text)

    words = text.split()

    words = [
        word.strip(".,!?;:()[]{}")
        for word in words
    ]

    words = [
        word
        for word in words
        if len(word) > 2
    ]

    words = words[:20]

    embeddings = create_embeddings(words)

    scores = calculate_attention(embeddings)

    st.subheader("🧠 Word Attention")

    display_scores = scores / scores.max()

    for word, score in zip(words, display_scores):

        st.write(f"**{word}**")

        st.progress(float(score))

    top_index = np.argmax(scores)
    top_word = words[top_index]

    st.success(
        f"⭐ Highest Attention: **{top_word}**"
    )