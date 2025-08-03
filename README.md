# PulseVision: Arrhythmia Classification Using CNN

[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**End-to-end CNN solution for classifying 5 types of heart arrhythmias from ECG signals with 95% accuracy**

---

## 📊 Performance Highlights
| Metric | Value | Significance |
|--------|-------|--------------|
| **Accuracy** | 95% | Reliable clinical-grade performance |
| **ROC-AUC** | 0.97 | Excellent class separation capability |
| **Inference Speed** | 5ms/ECG | Real-time processing capable |
| **Model Size** | 4.2 MB | Lightweight deployment |

---

## 🏗️ Project Architecture
```
PulseVision/
├── task_1.ipynb            # Main Jupyter notebook (training/evaluation)
├── cnn_ecg_model.h5        # Trained model weights
├── requirements.txt        # Python dependencies
├── confusion_matrix.png    # Visual classification errors
├── training_logs.csv       # Epoch-wise performance history
├── mitbih_train.csv        # Training dataset
└── mitbih_test.csv         # Testing dataset
```

---

## 🧠 Model Architecture
```python
model = Sequential([
    Conv1D(32, kernel_size=5, activation="relu", input_shape=(187,1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(5, activation="softmax")
])
```
**Optimization:**
- Adam optimizer (lr=0.001)
- Class-weighted loss function
- Early stopping with patience=5
- ReduceLROnPlateau with factor=0.5

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- GPU recommended for training

### Installation
```bash
# Create virtual environment
python -m venv pulse-env
source pulse-env/bin/activate  # Linux/MacOS
.\pulse-env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Execution
```bash
jupyter notebook task_1.ipynb
```
**Workflow:**
```mermaid
graph LR
A[Load CSV Data] --> B[Preprocessing]
B --> C[Class Weight Adjustment]
C --> D[CNN Model Training]
D --> E[Evaluation Metrics]
E --> F[Model Export]
```

---

## 🔍 Key Components

### Data Processing
```python
# Load MIT-BIH dataset
train_df = pd.read_csv("mitbih_train.csv", header=None)
test_df = pd.read_csv("mitbih_test.csv", header=None)

# Handle class imbalance
class_weights = compute_class_weight(
    "balanced", 
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))
```

### Training Configuration
```python
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
]

history = model.fit(
    X_train_reshaped, y_train_encoded,
    validation_data=(X_test_reshaped, y_test_encoded),
    epochs=50,
    batch_size=128,
    class_weight=class_weight_dict,
    callbacks=callbacks
)
```

---

## 📊 Results Analysis

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (N) | 0.99 | 0.99 | 0.99 | 18117 |
| 1 (S) | 0.92 | 0.85 | 0.88 | 556 |
| 2 (V) | 0.95 | 0.96 | 0.95 | 578 |
| 3 (F) | 0.94 | 0.90 | 0.92 | 641 |
| 4 (Q) | 0.97 | 0.99 | 0.98 | 1869 |

**Legend:**  
N = Normal, S = Supraventricular, V = Ventricular, F = Fusion, Q = Unknown

---

## 🧩 Output Files
| File | Purpose | Format |
|------|---------|--------|
| `cnn_ecg_model.h5` | Trained model weights | HDF5 |
| `confusion_matrix.png` | Visual classification errors | PNG |
| `training_logs.csv` | Epoch-wise metrics | CSV |
| `performance_report.txt` | Full classification metrics | Text |

---

## 📚 Dataset Information
**MIT-BIH Arrhythmia Database:**
- 109,446 ECG segments
- 5 heartbeat categories
- 187-point waveforms (360Hz sampling)
- Imbalanced distribution (mitigated via class weighting)

---

## 🔧 How to Improve
1. **Data Augmentation**  
   Add synthetic beats using GANs
2. **Model Optimization**  
   Experiment with ResNet1D architectures
3. **Real-time Integration**  
   Implement TensorRT conversion
4. **Deployment**  
   Create Flask API endpoint

---

## 👨‍💻 Author
**Pranav Bansode**  
[![Email](https://img.shields.io/badge/Email-pranavbansode2604%40gmail.com-blue)](mailto:pranavbansode2604@gmail.com)  
[![Colab](https://img.shields.io/badge/Colab-Notebook-orange)](https://colab.research.google.com/drive/1ypyA1d5j9Xp8gTzncXSkbDv1XxQ4XhOd?usp=sharing)
