
```markdown
# 🚀 HybridViT-ConvNeXt-GBC-Ultrasound-Classifier

An AI-powered Streamlit web application for **Gallbladder Cancer Detection** from ultrasound images using a **hybrid deep learning architecture** combining **ConvNeXt** and **Vision Transformer (ViT)**.

---

## 🔬 Project Overview

This tool classifies gallbladder ultrasound images into:
- 🟢 **Normal**
- 🟡 **Benign**
- 🔴 **Malignant**

It leverages the strengths of:
- 🧠 **ConvNeXt** for capturing **local features**
- 🔭 **Vision Transformer (ViT)** for modeling **global dependencies**

---

## 🎯 Key Features

- ✅ Upload ultrasound images and get predictions instantly
- ✅ Hybrid deep learning model (ConvNeXt + ViT)
- ✅ Clean, responsive interface using Streamlit
- ✅ PyTorch-based inference with optional GPU acceleration
- ✅ Unified frontend/backend workflow for researchers or clinicians

---

## 🛠️ Tech Stack

- **PyTorch** + **timm**
- **Vision Transformer (ViT)**
- **ConvNeXt**
- **Streamlit**
- **OpenCV** & **PIL**

---

## 🗂️ Project Structure
HybridViT-ConvNeXt-GBC-Ultrasound-Classifier/
│
├── app.py                          # Streamlit frontend UI
├── model/
│   └── NEW\_hybrid\_model\_finetuned.pth   # Pretrained hybrid model weights
├── utils/
│   └── preprocess.py               # Image preprocessing functions
├── requirements.txt                # Python dependencies
└── README.md

````

---

## 🚀 Getting Started

### 🔧 Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/HybridViT-ConvNeXt-GBC-Ultrasound-Classifier.git
cd HybridViT-ConvNeXt-GBC-Ultrasound-Classifier
````

### 🧪 Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 📦 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Step 4: Launch the Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Training Snapshot

| Epoch | Train Loss | Accuracy     |
| ----- | ---------- | ------------ |
| 1     | 5.1728     | 42.17%       |
| 2     | 3.0945     | 60.83%       |
| 3     | 1.9223     | 72.63%       |
| 4     | 1.1641     | 81.69%       |
| 5     | 0.8572     | 85.23%       |
| 6     | 0.6128     | 90.12%       |
| 7     | 0.3941     | 93.76%       |
| 8     | 0.2519     | 96.14%       |
| 9     | 0.1398     | 97.45%       |
| 10    | 0.0824     | **98.36%** ✅ |

---

## 📥 Example Usage

* Upload an ultrasound image (JPEG/PNG).
* The model will output the prediction: **Normal**, **Benign**, or **Malignant**.
* Fast, local, and privacy-respecting inference.

---

## 🤝 Let's Collaborate

This tool is ideal for:

* 🔬 Researchers in **Medical AI**
* 🏥 Clinicians looking for decision support
* 📚 Educators and students in **Computer Vision**

If you're working in **medical imaging**, **explainable AI**, or **deep learning for healthcare**, let’s connect!

---

## 📬 Contact

* 📧 Email: [Taponpaul250@gmail.com](mailto:Taponpaul250@gmail.com)
* 🌐 LinkedIn: [linkedin.com/in/tapon-paul-174267351](https://www.linkedin.com/in/tapon-paul-174267351/)

---

## 📜 License

This project is released under the [MIT License](LICENSE).

```

