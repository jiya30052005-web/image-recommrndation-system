import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Image Recommendation System", page_icon="🔍")
st.title("🔍 Image-Based Recommendation System")
st.write("Upload an image and get visually similar recommendations!")

@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()
    return model

@st.cache_data
def load_features():
    if os.path.exists("features.pkl"):
        with open("features.pkl", "rb") as f:
            return pickle.load(f)
    return {}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def extract_features(img, model):
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().numpy()

model = load_model()
features_dict = load_features()

if not features_dict:
    st.warning("⚠️ Dataset not found! Please run setup.py first.")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    if uploaded_file:
        query_img = Image.open(uploaded_file).convert("RGB")
        st.image(query_img, caption="Your Uploaded Image", width=250)
        query_features = extract_features(query_img, model)
        similarities = {}
        for name, feat in features_dict.items():
            sim = cosine_similarity([query_features], [feat])[0][0]
            similarities[name] = sim
        top3 = sorted(similarities, key=similarities.get, reverse=True)[:3]
        st.subheader("🎯 Top 3 Similar Images:")
        cols = st.columns(3)
        for i, img_name in enumerate(top3):
            with cols[i]:
                rec_img = Image.open(f"dataset/{img_name}")
                st.image(rec_img, caption=f"Score: {similarities[img_name]:.2f}", use_container_width=True)
            