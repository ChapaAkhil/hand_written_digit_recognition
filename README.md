# Handwritten Digit Recognition with TensorFlow & Keras

This project is a simple implementation of a handwritten digit recognition system using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. It uses the popular MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0–9).

---

## 🧠 Model Overview

- **Frameworks**: TensorFlow, Keras
- **Dataset**: MNIST (loaded via `tensorflow.keras.datasets`)
- **Model Type**: Convolutional Neural Network (CNN)
- **Architecture**:
  - Conv2D → ReLU → MaxPooling
  - Flatten
  - Dense → ReLU
  - Output Layer (Softmax)

---

## 📁 Files

- `hand_written_digit_recognition.ipynb` – The main Jupyter notebook containing the model code, training, testing, and predictions.

---

## 🔧 How to Run

1. **Install requirements** (if not already installed):

   ```bash
   pip install tensorflow matplotlib
````

2. **Run the notebook**:

   You can open and run the notebook using Jupyter:

   ```bash
   jupyter notebook
   ```

   Then navigate to `hand_written_digit_recognition.ipynb` and run all cells.

---

## 📈 Sample Output

After training, the model predicts the digit from a handwritten image. For example:

```python
np.argmax(y_pred[0])
```

Might return:

```
7
```

---

## 📊 Training Details

* **Epochs**: 5
* **Loss Function**: `sparse_categorical_crossentropy`
* **Optimizer**: `adam`
* **Metrics**: Accuracy

---

## ✅ Accuracy

The model achieves high accuracy on both training and test data, demonstrating the effectiveness of CNNs for image classification tasks like digit recognition.

---

## 🙌 Credits

* MNIST dataset by Yann LeCun
* TensorFlow and Keras libraries

---

