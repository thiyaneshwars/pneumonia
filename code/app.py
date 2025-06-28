# 🧠 Pneumonia Detection - CNN Model Training

This repository focuses on training a Convolutional Neural Network (CNN) to detect **pneumonia** from **chest X-ray images** using TensorFlow/Keras.

---

## 📁 Dataset Structure

Organize your dataset like this:

```

dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/

````

📌 Recommended dataset:  
👉 [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pneumonia-detector.git
cd pneumonia-detector
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train_model.py
```

✅ After training, the model will be saved as:

```
model/pneumonia_model.h5
```

---

## 🧠 Model Architecture

* Input size: `150x150x3`
* CNN Layers:

  * Conv2D → ReLU → MaxPooling
  * Conv2D → ReLU → MaxPooling
  * Flatten → Dense → Dropout → Sigmoid
* Output: Binary Classification (`NORMAL` / `PNEUMONIA`)
* Optimizer: Adam
* Loss Function: Binary Crossentropy

---

## 🔧 Configuration

Adjust settings in `train_model.py`:

* Image size
* Batch size
* Epochs
* Data paths

---

## ✅ Requirements

```
tensorflow
numpy
opencv-python
```

Install with:

```bash
pip install tensorflow numpy opencv-python
```

---

## 👨‍💻 Author

**Thiyaneshwar S**
🎓 B.E. in AIML | 🤖 AI Developer | 🛡️ Ethical Hacking Enthusiast
📧 Email: [sthiyaneshwar94@gmail.com](mailto:sthiyaneshwar94@gmail.com)

---

## 📄 License

This code is released for academic and educational purposes only.



