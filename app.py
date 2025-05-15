import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

IMG_SIZE = (224, 224)

CLASS_LABELS = {
    "apple": ['Alteria leaf spot', 'Brown spot', 'Gray spot', 'Healthy leaf', 'Rust'],
    "banana": ['cordana', 'healthy', 'pestalotiopsis', 'sigatoka'],
    "corn": ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy'],
    "cotton": ['bacterial_blight', 'curl_virus', 'fussarium_wilt', 'healthy'],
    "mango": ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge',
              'Healthy', 'Powdery Mildew', 'Sooty Mould'],
    "potato": ["Early Blight", "Late Blight", "Healthy"],
    "rice": ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast',
             'leaf_scald', 'narrow_brown_spot'],
    "sugarcane": ['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow'],
    "tomato": ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
               'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'],
}

FERTILIZER_SUGGESTIONS = {
    ("apple", "Alteria leaf spot"): "Apply Mancozeb and maintain soil pH around 6.5.",
    ("apple", "Brown spot"): "Use Captan or copper-based fungicides. Maintain proper spacing.",
    ("apple", "Gray spot"): "Use broad-spectrum fungicide and ensure good drainage.",
    ("apple", "Rust"): "Apply sulfur-based fungicides and avoid excess nitrogen.",

    ("banana", "cordana"): "Use Thiram fungicide and improve potassium intake.",
    ("banana", "pestalotiopsis"): "Apply copper oxychloride and organic manure.",
    ("banana", "sigatoka"): "Use a balanced NPK fertilizer (19-19-19) and apply Dithane M-45.",
    
    ("corn", "Blight"): "Apply potassium-rich fertilizers and avoid nitrogen excess.",
    ("corn", "Common_Rust"): "Use Dithane M-45 and phosphorus-rich fertilizers.",
    ("corn", "Gray_Leaf_Spot"): "Improve zinc and potash application.",
    
    ("cotton", "bacterial_blight"): "Spray copper-based bactericides and reduce nitrogen fertilizer.",
    ("cotton", "curl_virus"): "Use boron and magnesium fertilizers and remove infected plants.",
    ("cotton", "fussarium_wilt"): "Use organic compost and apply Trichoderma.",
    
    ("mango", "Anthracnose"): "Use Bordeaux mixture and nitrogen-phosphorus-potassium fertilizer.",
    ("mango", "Bacterial Canker"): "Spray Streptomycin and use compost-rich manure.",
    ("mango", "Cutting Weevil"): "Use neem oil and potassium nitrate spray.",
    ("mango", "Die Back"): "Apply copper oxychloride and balance NPK dose.",
    ("mango", "Gall Midge"): "Use systemic insecticide and organic mulch.",
    ("mango", "Powdery Mildew"): "Spray wettable sulfur and increase phosphorus.",
    ("mango", "Sooty Mould"): "Control pests first and spray neem oil.",
    
    ("potato", "Early Blight"): "Apply Mancozeb and increase potash levels.",
    ("potato", "Late Blight"): "Apply Ridomil Gold and use MOP for potassium.",
    
    ("rice", "bacterial_leaf_blight"): "Use potash, avoid urea, and apply Streptocycline.",
    ("rice", "brown_spot"): "Use zinc sulphate and increase potash application.",
    ("rice", "leaf_blast"): "Spray tricyclazole and apply phosphorus-based fertilizer.",
    ("rice", "leaf_scald"): "Use potash-rich fertilizer and maintain field hygiene.",
    ("rice", "narrow_brown_spot"): "Improve soil pH and use balanced NPK.",
    
    ("sugarcane", "RedRot"): "Apply potash, and avoid excess nitrogen.",
    ("sugarcane", "Mosaic"): "Use virus-resistant varieties and avoid excess nitrogen.",
    ("sugarcane", "Rust"): "Apply fungicides like Mancozeb and increase potassium.",
    ("sugarcane", "Yellow"): "Use balanced fertilizers with micronutrients (Zn, Fe).",
    
    ("tomato", "Tomato___Bacterial_spot"): "Spray copper fungicide and use potassium sulfate.",
    ("tomato", "Tomato___Early_blight"): "Use chlorothalonil and phosphorus-rich fertilizers.",
    ("tomato", "Tomato___Late_blight"): "Spray metalaxyl and apply MOP.",
    ("tomato", "Tomato___Leaf_Mold"): "Spray fungicide like Chlorothalonil and use compost-rich organic manure.",
    ("tomato", "Tomato___Septoria_leaf_spot"): "Apply mancozeb and ensure nitrogen is balanced.",
    ("tomato", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"): "Spray neem oil and use micronutrient mix.",
    ("tomato", "Tomato___Tomato_mosaic_virus"): "Remove infected plants and apply potassium nitrate.",
}


st.set_page_config(page_title="Plant Disease Detection", layout="wide")

# Initialize session state variables
if "started" not in st.session_state:
    st.session_state.started = False
if "selected_crop" not in st.session_state:
    st.session_state.selected_crop = None
if "model" not in st.session_state:
    st.session_state.model = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None


def load_crop_model(crop_name):
    model_dir = "models"  # Adjust this path if needed
    model_path = f"{model_dir}/{crop_name.lower()}_disease_model.h5"
    return load_model(model_path)


def predict_image(model, image):
    image_resized = image.resize(IMG_SIZE)
    image_array = img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    preds = model.predict(image_array)
    pred_class_idx = np.argmax(preds, axis=1)[0]
    confidence = preds[0][pred_class_idx]
    return pred_class_idx, confidence


# Start Page
if not st.session_state.started:
    st.title("🌿 Welcome to Plant Disease Prediction")
    st.write("Click the button below to begin.")
    if st.button("Start"):
        st.session_state.started = True
    st.stop()

# Crop Selection
if st.session_state.selected_crop is None:
    st.title("🌾 Select a Crop")
    crops = list(CLASS_LABELS.keys())
    crop_choice = st.selectbox("Choose the crop you want to analyze:", [c.capitalize() for c in crops])
    if st.button("Continue"):
        crop_key = crop_choice.lower()
        st.session_state.selected_crop = crop_key
        try:
            st.session_state.model = load_crop_model(crop_key)
            st.success(f"Model for **{crop_choice}** loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load model for {crop_choice}. Error: {e}")
            st.session_state.selected_crop = None
    st.stop()

# Image upload and prediction
st.title(f"🌿 Crop selected: {st.session_state.selected_crop.capitalize()}")

uploaded_file = st.file_uploader("Upload a leaf or plant image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.session_state.uploaded_image = image

    if st.button("Predict"):
        if st.session_state.model is not None and st.session_state.uploaded_image is not None:
            pred_idx, conf = predict_image(st.session_state.model, st.session_state.uploaded_image)
            crop = st.session_state.selected_crop
            labels = CLASS_LABELS[crop]
            pred_label = labels[pred_idx] if pred_idx < len(labels) else "Unknown"
            st.session_state.prediction = pred_label
            st.session_state.confidence = conf
        else:
            st.error("Model or image not loaded correctly.")

# Show prediction results
if st.session_state.prediction:
    st.markdown(f"### 🧪 Prediction: **{st.session_state.prediction}**")
    st.markdown(f"### 📊 Confidence: **{st.session_state.confidence*100:.2f}%**")

    # Fertilizer Suggestion
    crop = st.session_state.selected_crop
    disease = st.session_state.prediction.lower()
    key = (crop, disease.lower())
    fertilizer_tip = FERTILIZER_SUGGESTIONS.get((crop, st.session_state.prediction), None)

    if fertilizer_tip and "healthy" not in disease:
        st.markdown("### 🌱 Recommended Fertilizer / Treatment")
        st.success(fertilizer_tip)
    elif "healthy" in disease:
        st.info("🟢 Your crop looks healthy! No treatment is needed.")

# Option to restart
if st.button("🔁 Start Over"):
    for key in ["started", "selected_crop", "model", "uploaded_image", "prediction", "confidence"]:
        st.session_state[key] = None
    st.session_state.started = False
    st.experimental_rerun()
