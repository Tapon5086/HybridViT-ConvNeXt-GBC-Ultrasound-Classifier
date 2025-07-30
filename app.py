import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import timm
from PIL import Image
import numpy as np

# Set title
st.title("GBCU Hybrid Model Predictor")
st.markdown("Upload a Ultrasound image and click **Predict** to classify it as Normal, Benign, or Malignant.")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Model Components ===
def get_convnext_model():
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 3)
    return model

def get_vit_model(num_classes=3):
    model = timm.create_model('vit_base_patch32_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

class HybridModel(nn.Module):
    def __init__(self, num_classes=3):
        super(HybridModel, self).__init__()
        self.convnext = get_convnext_model().to(device)
        self.vit = get_vit_model(num_classes).to(device)

        self.convnext.classifier = nn.Identity()
        self.vit.head = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        convnext_features = self.convnext(x).flatten(start_dim=1)
        vit_features = self.vit(x)
        combined_features = torch.cat((convnext_features, vit_features), dim=1)
        return self.fc(combined_features)

# Load model
@st.cache_resource
def load_model():
    model = HybridModel(num_classes=3).to(device)
    model.load_state_dict(torch.load('C:/Users/ASUS/Dropbox/PC/Desktop/GBC_Detection/hybrid_model_finetuned.pth', map_location=device))

    model.eval()
    return model

model = load_model()
class_names = ['Normal-0', 'Benign-1', 'Malignant-2'] 

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Predict button
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            img_tensor = transform(image).unsqueeze(0).to(device)
            outputs = model(img_tensor)
            _, pred = torch.max(outputs, 1)
            predicted_class = class_names[pred.item()]

        st.success(f"Prediction: **{predicted_class}**")

