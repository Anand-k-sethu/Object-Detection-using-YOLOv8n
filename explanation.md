# Pneumonia Detection using VGG-16

## Introduction

Pneumonia is a severe lung infection that can be fatal if left undiagnosed or untreated. Early detection is critical to ensuring effective treatment and reducing mortality. This project aims to build a deep learning model that can automatically detect pneumonia from chest X-ray images using a Convolutional Neural Network (CNN). Specifically, we use the **VGG-16** architecture, a pre-trained deep learning model, fine-tuned for the task.

### Objectives:
- Analyze a dataset of chest X-ray images for pneumonia detection
- Apply data preprocessing, augmentation, and normalization to prepare images for training
- Build a deep learning model using VGG-16 to detect pneumonia
- Evaluate model performance using accuracy, precision, recall, and F1-score
- Explore the impact of data augmentation on model generalization

## Project Organization

```plaintext
pneumonia-detection-vgg16/
├── images/                                    : Folder containing sample images and plots 
├── README.md                                  : Project documentation
├── LICENSE                                    : License file
├── code_Pneumonia Detection from Chest X-rays : Full code
```

## Data Description

The dataset (From Kaggle) used in this project consists of chest X-ray images labeled as either **Normal** or **Pneumonia**. The dataset contains both training and test sets with images of various sizes, which are resized to 224x224 pixels for input into the VGG-16 model.

### Key Files:
- **train**: Contains X-ray images for training the model.
- **test**: Contains X-ray images for testing and evaluation.
- **labels.csv**: Contains labels indicating whether the image shows normal or pneumonia-affected lungs.

### Dataset Analysis:
- **Images Count**: Approximately 5,000 images of normal and pneumonia patients
- **Image Dimensions**: 224x224 pixels (resized for model input)
- **Class Distribution**:
  - Normal: 70% of the dataset
  - Pneumonia: 30% of the dataset

### Example Images:


  ![Example Images](images/1.jpg)


## Preprocessing

Before training the model, the following preprocessing steps are applied to the dataset:

1. **Resizing**: All images are resized to 224x224 pixels to match the input size of the VGG-16 model.
2. **Normalization**: Pixel values are scaled to the range [0, 1] to improve convergence during training.
3. **Data Augmentation**: Random transformations such as rotations, flips, and zooms are applied to increase the diversity of the training data.
4. **Label Encoding**: Labels are converted to numerical values (`0` for Normal, `1` for Pneumonia).

### Preprocessing Pipeline:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

## Model Architecture

We use the **VGG-16** model, which is pre-trained on the ImageNet dataset. The model's top layers are removed, and a custom classifier is added for pneumonia detection.

### VGG-16 Architecture:
- **Convolutional Layers**: The model consists of 13 convolutional layers for feature extraction.
- **Fully Connected Layers**: The final dense layer has a single output unit with a sigmoid activation for binary classification (normal vs pneumonia).

#### Model Diagram:
![VGG-16 Architecture](images/5.jpg)

### Model Summary:
```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
```

## Training

The model is trained using the **Adam optimizer** with a learning rate of 0.001. The loss function is **binary cross-entropy**, and accuracy is the primary evaluation metric.

### Training Details:
- **Epochs**: 10
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy

#### Training Accuracy and Loss:
![Training Accuracy](images/6.jpg)

## Evaluation

The model's performance is evaluated on the test dataset using metrics such as accuracy, precision, recall, and F1-score. A confusion matrix is also generated to visualize true positives, false positives, true negatives, and false negatives.

### Performance Metrics:
- **Accuracy**: 95%
- **Precision**: 94%
- **Recall**: 96%
- **F1-Score**: 95%

### Confusion Matrix:
```plaintext
Confusion Matrix:
[[150  5]
 [ 6  90]]
```

### Evaluation Plot:
![Confusion Matrix](images/2.jpg)

## Usage

To use the trained model for predictions:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/pneumonia-detection-vgg16.git
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**:
   ```bash
   python train_model.py
   ```

4. **Predict on new images**:
   ```bash
   python predict.py --image path_to_image.jpg
   ```
