# Dog vs Cat Classification Using MobileNet

This project demonstrates how to classify images of dogs and cats using transfer learning with MobileNetV2 from TensorFlow Hub. The MobileNetV2 model is a powerful convolutional neural network optimized for mobile and embedded vision applications.

## Overview

- **Dataset**: The dataset used is from the [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats) Kaggle competition.
- **Model**: MobileNetV2 is used as a feature extractor to classify images of dogs and cats.
- **Technologies**: TensorFlow, TensorFlow Hub, Keras, Python, Jupyter Notebook.

### Project Steps

1. **Data Acquisition**: 
   - The dataset was downloaded from Kaggle using the Kaggle API.
   - The dataset consists of images of dogs and cats.

2. **Data Preprocessing**:
   - Extracted images from the compressed `.zip` files.
   - Displayed some sample images from the dataset to visualize the content.
   - Resized all images to `(224, 224)` to match the expected input size of MobileNetV2.

3. **Label Assignment**:
   - Created labels for the images: `1` for dogs and `0` for cats.

4. **Feature Extraction**:
   - Converted images into NumPy arrays for model training.
   - Split the dataset into training and testing sets (80% training, 20% testing).
   - Scaled the image data to range `[0, 1]` for better model performance.

5. **Model Training**:
   - Built a neural network using MobileNetV2 as a feature extractor with a dense output layer for classification.
   - Compiled the model with the Adam optimizer and Sparse Categorical Crossentropy loss.
   - Trained the model for 5 epochs with promising results.

6. **Results**:
   - Achieved high accuracy during training and evaluation.
   - Test accuracy reached around `97.75%` after 5 epochs.

### Dataset

- **Training Data**: 10,000 images of dogs and cats resized to `(224, 224)`.
- **Labels**: Binary labels (`0` for cat, `1` for dog`).

### Model Details

- **Pre-trained Model**: MobileNetV2 (`https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4`) from TensorFlow Hub.
- **Transfer Learning**: Used MobileNetV2 for extracting image features and added a dense layer with 2 units to classify the images.

### Key Technologies

- **TensorFlow Hub**: Pre-trained model integration.
- **Pandas, NumPy, Matplotlib**: Data handling and visualization.
- **Keras**: Model building and training.

### Results

The model achieves a training accuracy of over `99%` and a testing accuracy of `97.75%`. This demonstrates the effectiveness of MobileNetV2 in feature extraction for binary image classification tasks.

### Sample Results

The trained model was able to classify the test images of dogs and cats with a high accuracy as shown below:

- **Test Loss**: 0.0812
- **Test Accuracy**: 97.75%

### Images Used for Training

Sample images of dogs and cats used for training:

- **Dog Image**: 
  ![Dog Image](dog_sample_image_path)
- **Cat Image**:
  ![Cat Image](cat_sample_image_path)


### References

- [MobileNetV2 on TensorFlow Hub](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4)
- [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats)
