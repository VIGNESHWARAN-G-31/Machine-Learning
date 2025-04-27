# Line Angle Classification using CNN 

## Project Overview

This project focuses on generating synthetic images of lines at different angles (0°, 45°, 90°, and 135°) and training a Convolutional Neural Network (CNN) to classify them correctly.

All data is stored and accessed via **Google Drive**. The model is trained using TensorFlow and Keras, and its performance is evaluated with multiple metrics.

---

## Setup Instructions 🚀

1. **Environment**:

   - Google Colab Notebook
   - Python 3.x
   - TensorFlow 2.x
   - Matplotlib, NumPy, Scikit-learn

2. **Libraries Required**:

   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   import tensorflow as tf
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import classification_report, confusion_matrix
   import os
   import cv2
   import random
   ```

3. **Connect to Google Drive**:

   - Mount Google Drive to Colab to save datasets and outputs.
   - Link: [Drive Folder](https://drive.google.com/drive/folders/1dE_-f4-DYJ3KVpC9jzqFlD_9o-oaD5SH)

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

---

## Execution Steps 🔥

### 1. Create Datasets Folder

Create a new folder named `datasets` inside the Google Drive directory to store synthetic images.

### 2. Generate Synthetic Images

- **Function to generate random background color**.
- **Function to draw lines based on the given angle** (0°, 45°, 90°, 135°).
- **Save images** under corresponding folders (`0`, `45`, `90`, `135`) in the datasets directory.

### 3. Load the Dataset

- Set the dataset folder path.
- Read images using `cv2.imread()`.
- Resize and normalize images for consistency.

### 4. Prepare the Data

- **Normalization**:\
  Pixel values are scaled from [0, 255] to [0, 1] to improve model performance.

- **Visualization**:\
  Show before and after normalization to understand the impact.

- **One-Hot Encoding**:\
  Labels (0°, 45°, 90°, 135°) are one-hot encoded to be suitable for CNN classification.

- **Visualization after encoding**:\
  Sample one-hot encoded labels are visualized.

- **Train-Test Split**:\
  Split the data into 80% training and 20% testing.

### 5. Build the CNN Model

- Sequential model with layers:
  - Conv2D
  - MaxPooling2D
  - Flatten
  - Dense layers

Example architecture:

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(image_size, image_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])
```

### 6. Train the CNN Model

- Compile using `adam` optimizer and `categorical_crossentropy` loss.
- Fit the model on training data and validate on test data.
- Set epochs and batch size as required.

### 7. Evaluate the Model

- Generate a **classification report** (precision, recall, f1-score).
- Display a **confusion matrix**.
- Plot **accuracy and loss** curves over epochs.

Example to evaluate:

```python
evaluation = model.evaluate(X_test, y_test, verbose=2)
test_loss = evaluation[0]
test_acc = evaluation[1]
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
```

### 8. Predict Individual Images

- Test prediction on random test samples.
- Compare **Predicted Angle vs True Angle** visually.

Example visualization:

```python
plt.figure(figsize=(10, 6))

for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[i])
    plt.title(f"Pred: {angles[np.argmax(y_pred[i])]}
True: {angles[np.argmax(y_test[i])]}")
    plt.axis('off')

plt.show()
```

---

## Result Interpretation 📊

- **Test Accuracy**:\
  Achieved after evaluating on the unseen test dataset. A good indicator of model generalization.

- **Test Loss**:\
  Represents how well the model minimizes the classification error.

- **Classification Report**:\
  Shows detailed metrics for each class (0°, 45°, 90°, 135°).

- **Confusion Matrix**:\
  Shows how many images were correctly/incorrectly classified for each angle.

- **Visual Predictions**:\
  Helps in visually understanding where the model is predicting correctly and where it might confuse between angles.

---

## Project Folder Structure 📂

```
/drive/MyDrive/
    └── datasets/
        ├── 0/
        ├── 45/
        ├── 90/
        └── 135/
    └── Colab_Notebook.ipynb
```

---

## Final Notes ✍️

- Make sure to re-run **Drive mount** if the session disconnects.
- You can expand to more angles or make background noise more complex for a more challenging dataset.
- Tune the CNN architecture (more layers, regularization) to improve accuracy further if needed.

---

give me the downloadable format\


