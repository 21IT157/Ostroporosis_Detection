import streamlit as st
import torch
import numpy as np
from torchvision import transforms
from transformers import BertTokenizer, BertForSequenceClassification
from PIL import Image

# Load the pre-trained BERT model for sequence classification (adjust this if your model is different)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load your custom model weights
state_dict = torch.load('/workspaces/Ostroporosis_Detection/bert_model.pt', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Initialize the tokenizer for BERT (this is important for text-based input if you're using BERT for text)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the transformation for input data (this is for image-based models, if applicable)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to the required input size
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize if needed
])

def predict_image(img):
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(img_tensor)
    prediction = torch.argmax(output, dim=1).item()
    return "Osteoporosis" if prediction == 1 else "No Osteoporosis"

def predict_osteoporosis(features):
    # Assuming features is a string (like a health report or description)
    inputs = tokenizer(features, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    
    return "Osteoporosis" if prediction == 1 else "No Osteoporosis"

# Streamlit app layout
st.title("Osteoporosis Prediction Form")
st.write("Please fill in the details below:")

# Input fields for various health-related details (features for the osteoporosis model)
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", options=["Male", "Female"])
hormonal_changes = st.selectbox("Hormonal Changes", options=["Normal", "Postmenopause"])
family_history = st.selectbox("Family History", options=["Yes", "No"])
race = st.selectbox("Race", options=["Asian", "Caucasian", "African American"])
body_weight = st.selectbox("Body Weight", options=["Underweight", "Normal"])
calcium_intake = st.selectbox("Calcium Intake", options=["Low", "Adequate"])
vitamin_d_intake = st.selectbox("Vitamin D Intake", options=["Sufficient", "Insufficient"])
physical_activity = st.selectbox("Physical Activity", options=["Sedentary", "Active"])
smoking = st.selectbox("Smoking", options=["Yes", "No"])
alcohol_consumption = st.selectbox("Alcohol Consumption", options=["Moderate", "NaN"])
medical_conditions = st.selectbox("Medical Conditions", options=["None", "Hyperthyroidism", "Rheumatoid Arthritis"])
medications = st.selectbox("Medications", options=["Corticosteroids", "None"])
prior_fractures = st.selectbox("Prior Fractures", options=["Yes", "No"])

# Button to submit the form
if st.button("Predict"):
    # Collect all features into a list and convert to a text input
    features = f"Age: {age}, Gender: {gender}, Hormonal Changes: {hormonal_changes}, Family History: {family_history}, Race: {race}, Body Weight: {body_weight}, Calcium Intake: {calcium_intake}, Vitamin D Intake: {vitamin_d_intake}, Physical Activity: {physical_activity}, Smoking: {smoking}, Alcohol Consumption: {alcohol_consumption}, Medical Conditions: {medical_conditions}, Medications: {medications}, Prior Fractures: {prior_fractures}"
    
    # Make prediction using the BERT model for osteoporosis risk based on features
    result = predict_osteoporosis(features)
    
    # Display prediction result
    st.write(f"Prediction: **{result}**")
