# Rice Variety Classification using Machine Learning

This project, developed for a Machine Learning course at Universiti Putra Malaysia, automates the classification of five different rice varieties (Arborio, Basmati, Ipsala, Jasmine, and Karacadag) using deep learning and traditional neural networks.

## Problem Statement
Traditional methods for classifying rice varieties are labor-intensive, time-consuming, and prone to human error. This project leverages image processing and machine learning to create an efficient, accurate, and scalable classification system to aid in agricultural quality control.

## Dataset
The model was trained on a public dataset containing 75,000 images (15,000 for each of the 5 varieties). A comprehensive feature dataset was extracted from these images, resulting in **106 features** for each sample, including:
* 12 Morphological features (e.g., Area, Perimeter, Eccentricity)
* 4 Shape features
* 90 Color features (derived from RGB, HSV, and L*a*b* color spaces)

## Methodology
1.  **Data Preprocessing:** The dataset was cleaned, normalized, and standardized. Data augmentation techniques (rotation, zoom, flip) were applied to the image data to improve model generalization.
2.  **Model Building:** Two distinct neural network architectures were built, trained, and evaluated:
    * **Artificial Neural Network (ANN):** A multi-layer perceptron trained on the 106-feature dataset.
    * **Convolutional Neural Network (CNN):** A deep learning model trained directly on the rice grain images (250x250 pixels).
3.  **Evaluation:** Models were assessed using key classification metrics, including Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

## Technologies Used
* **Language:** Python
* **Libraries:**
    * TensorFlow & Keras (for building ANN and CNN models)
    * Scikit-learn (for data splitting and evaluation metrics)
    * Pandas (for handling the feature dataset)
    * NumPy (for numerical operations)
    * Matplotlib/Seaborn (for visualization)
