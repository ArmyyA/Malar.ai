# Malar.ai - Malaria Detection Using Modified ResNet50

This project aims to detect malaria from cell images using a modified version of the ResNet50 architecture to process grayscale images.

## Overview

Malaria is a life-threatening disease caused by parasites transmitted to humans through the bites of infected female Anopheles mosquitoes. Early and accurate detection is crucial for effective disease management.

In this project, I adapt the ResNet50 architecture, pre-trained on ImageNet, to process grayscale images for malaria detection.

## Features

- **Data Processing**: Images are loaded, resized to 224x224 pixels, and converted to grayscale.
- **Customized ResNet50**: I have modified the ResNet50 architecture to accept single-channel (grayscale) images. The single channel is then expanded to mimic a 3-channel image to utilize the ResNet50 architecture more effectively.
- **Normalization**: Data is normalized based on the average of the ImageNet RGB means to align with the expectations of the pre-trained weights.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- Keras
- TensorFlow
- scikit-learn
- Cell Dataset, which can be found [here](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/ArmyyA/Malar.ai.git
   cd Malar.ai
   ```

2. Place your dataset in the appropriate directory (`malaria-dataset/positive` for positive samples and `malaria-dataset/negative` for negative samples).

3. Run the script:

   ```bash
   python main.py
   ```

4. The trained model will be saved as `malaria_detection_model.h5`.

## Metrics and Results

The model performance was evaluated using the following metrics:

- **Loss**: Binary Crossentropy
- **Accuracy**: Percentage of correctly classified images

**Training Results**:

- Prediction Accuracy: 97%
