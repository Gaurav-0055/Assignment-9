

# 🐱 vs 🐶 Image Classification using CNN

### Assignment 9 — AI/ML

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen)]()
[![License](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/salader/dogs-vs-cats)

</div>

---

## 👤 Student Details

| Field | Details |
|---|---|
| 🧑 **Name** | Gaurav Gour |
| 🆔 **Registration No.** | 23BSA10096 |
| 📄 **Application No.** | IN26011516 |
| 🎓 **Batch No.** | 1A |
| 📚 **Assignment** | Assignment 9 — Image Classification using CNN |

---

## 🎯 Objective

Build a **Convolutional Neural Network (CNN)** to automate the classification of pet images into **Cats** and **Dogs**, for an animal welfare organization looking to streamline image-based pet identification. The project covers the complete ML pipeline — data understanding, preprocessing, model building, training, and evaluation.

---

## 📊 Dataset Link

> **Cats vs Dogs Dataset (Kaggle)**
> 🔗 https://www.kaggle.com/datasets/salader/dogs-vs-cats

⚠️ *The dataset was downloaded and uploaded manually to Google Colab. It is **not included** in this repository per submission guidelines — use the Kaggle link above.*

---

## 🧰 Libraries Used

| Library | Purpose |
|---|---|
| 🧠 `tensorflow` / `keras` | Model building, training, data generators |
| 🔢 `numpy` | Numerical operations |
| 📈 `matplotlib` | Plotting accuracy/loss graphs and sample images |
| 🎨 `seaborn` | Confusion matrix heatmap |
| 📏 `scikit-learn` | Precision, recall, F1-score, confusion matrix |
| 🖼️ `Pillow (PIL)` | Image verification and corrupted-file cleanup |
| 📁 `os`, `zipfile` | File handling and dataset extraction |

---

## 🔬 Methodology

| Step | Description |
|---|---|
| **1. Data Understanding** | Loaded the dataset, inspected folder structure (`PetImages/Cat`, `PetImages/Dog`), displayed sample images with labels, identified class counts and image dimensions |
| **2. Data Cleaning** | Removed corrupted/unreadable image files that would break the data generator during training |
| **3. Data Preprocessing** | Resized all images to **128×128** pixels, normalized pixel values to **0–1**, split into **80% train / 20% test** using `ImageDataGenerator` |
| **4. Model Development** | Built a CNN with three Conv2D + MaxPooling blocks, followed by Flatten, Dense, and a sigmoid output layer |
| **5. Training** | Compiled with Adam optimizer + binary crossentropy loss, trained for **10 epochs** |
| **6. Evaluation** | Assessed on the test set using accuracy, precision, recall, F1-score, confusion matrix, and accuracy/loss curves |

---

## 🏗️ CNN Architecture

| # | Layer | Configuration |
|---|---|---|
| 1️⃣ | Conv2D | 32 filters, 3×3 kernel, ReLU |
| 2️⃣ | MaxPooling2D | 2×2 pool size |
| 3️⃣ | Conv2D | 64 filters, 3×3 kernel, ReLU |
| 4️⃣ | MaxPooling2D | 2×2 pool size |
| 5️⃣ | Conv2D | 128 filters, 3×3 kernel, ReLU |
| 6️⃣ | MaxPooling2D | 2×2 pool size |
| 7️⃣ | Flatten | — |
| 8️⃣ | Dense | 128 neurons, ReLU |
| 9️⃣ | Output (Dense) | 1 neuron, Sigmoid |

**⚙️ Compilation Settings**

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Metric | Accuracy |
| Epochs | 10 |

---

## 📈 Results

### ✅ Test Metrics

| Metric | Score |
|---|---|
| 🎯 Test Accuracy | **83.29%** |
| 🎯 Precision | **85.59%** |
| 🎯 Recall | **80.07%** |
| 🎯 F1-Score | **82.74%** |

### 🔢 Confusion Matrix

| | Predicted: Cat 🐱 | Predicted: Dog 🐶 |
|---|---|---|
| **Actual: Cat 🐱** | 2162 ✅ | 337 ❌ |
| **Actual: Dog 🐶** | 498 ❌ | 2001 ✅ |

### 🧾 Observations

| # | Observation |
|---|---|
| 1️⃣ | **Overfitting is visible** — training accuracy climbed to ~98.7% while validation accuracy plateaued around 83%. Validation loss bottomed out at epoch 3 (~0.37) before rising sharply to ~0.83 by epoch 8, showing the model memorized training data past epoch 3–4. |
| 2️⃣ | **Solid but imperfect performance** — 83.29% test accuracy with precision (85.59%) higher than recall (80.07%), meaning the model is slightly conservative in predicting "Dog." |
| 3️⃣ | **Class-wise error asymmetry** — 337 cats misclassified as dogs vs. 498 dogs misclassified as cats (~50% more errors), suggesting dog images have greater visual variability (breeds, poses, fur patterns). |
| 4️⃣ | **Early stopping would help** — since validation loss was best around epoch 3–4, training the full 10 epochs without early stopping/regularization caused unnecessary overfitting later on. |

---

## 🏁 Conclusion

This project developed a **Convolutional Neural Network (CNN)** to classify images of cats and dogs, achieving a test accuracy of **83.29%**, with a precision of **85.59%**, recall of **80.07%**, and an F1-score of **82.74%**. The model learned meaningful visual features quickly, reaching over 80% validation accuracy within the first two epochs, though training beyond epoch 3–4 led to overfitting, as seen in the diverging train/validation loss curves.

**Convolution layers** are central to this performance — they automatically extract spatial features (edges, textures, shapes) directly from raw pixel data, removing the need for manual feature engineering. **Pooling layers** then reduce the spatial dimensions of these feature maps, lowering computational cost and making learned features more robust to small translations and distortions.

Compared to a standard **Artificial Neural Network (ANN)**, a CNN's key advantage for image classification is its ability to preserve and exploit spatial relationships between pixels — ANNs flatten images into 1D vectors and lose this structure, resulting in far more parameters and weaker performance on visual data.

One **limitation** of CNNs, evident in this project, is their tendency to overfit when trained for too many epochs without regularization — the model reached near-perfect training accuracy (98.7%) while validation accuracy stagnated around 83%, showing that techniques like dropout, data augmentation, or early stopping are needed to generalize well to unseen images.

---


