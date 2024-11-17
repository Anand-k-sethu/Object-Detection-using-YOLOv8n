# YOLO Object Detection with Pascal VOC Dataset

## Project Overview

In this project, the **PASCAL VOC 2012** dataset is used to train a **YOLOv8** object detection model. This model is capable of identifying and classifying various objects in real-time, such as cars, animals, and people. The goal is to build a system that can automatically detect and classify objects in images, which can be applied in fields like surveillance, autonomous driving, and industrial automation.

### Why This Project Matters

Object detection is a key technology for many real-world applications, from **self-driving cars** to **security systems**. By training a model to detect objects in images, you can automate the process, reduce human error, and improve system efficiency. This project provides the foundation for developing such systems, making it possible to identify objects quickly and accurately in a variety of scenarios.

## Key Features

- **Data Preprocessing**: Converts the PASCAL VOC dataset annotations into the YOLO format, which is optimized for training object detection models.
- **Dataset Subsetting**: Selects random subsets of the dataset for training and validation, ensuring that the model sees a diverse set of examples.
- **Model Training**: Utilizes the YOLOv8 architecture to train a model that can detect 20 different object classes, ranging from animals to vehicles.
- **Performance Evaluation**: After training, the model's performance is measured using **Precision**, **Recall**, and **mAP (Mean Average Precision)** to evaluate how well it detects and classifies objects.

## Skills and Technologies Used

- **Python**: The core language for data processing and model training.
- **Machine Learning & Deep Learning**: Implementing the YOLO architecture for object detection.
- **Data Processing**: Converting and preparing data to make it compatible with the YOLO format.
- **Model Evaluation**: Using performance metrics to evaluate the model’s effectiveness.

## Steps Involved

### 1. Convert PASCAL VOC Annotations to YOLO Format

The first step is to convert the dataset's annotations, which are originally in XML format (PASCAL VOC), into the YOLO format. In the YOLO format, each object is represented by a `.txt` file where each line contains the object’s class ID and normalized bounding box coordinates (center_x, center_y, width, height). This is essential for training the YOLO model.

```python
def convert_voc_to_yolo(voc_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over XML annotation files and convert them to YOLO format
    for xml_file in os.listdir(os.path.join(voc_dir, 'Annotations')):
        # Parsing XML file and converting annotations
        ...
```

### 2. Select Random Subsets for Training and Validation

To make sure the model can generalize well, random subsets of images and annotations are selected for training and validation. This helps ensure that the model is exposed to a wide variety of examples, preventing overfitting.

```python
def select_random_subset(list_file, subset_size):
    with open(list_file, 'r') as f:
        file_list = f.read().strip().split()
    random.shuffle(file_list)
    return file_list[:subset_size]
```

### 3. Copy Subset Files to New Directories

Once the subsets are selected, the corresponding image and annotation files are copied to new directories dedicated to training and validation. This structure keeps the dataset organized for easier use during training.

```python
def copy_files_from_list(file_list, source_img_dir, source_lbl_dir, dest_img_dir, dest_lbl_dir):
    # Copies image and label files to their respective directories
    ...
```

### 4. Count Class Distribution

Before training, it's crucial to ensure that the dataset has a balanced class distribution. By counting how many instances of each object class exist in the training and validation sets, we can identify any imbalances that might affect model performance.

```python
def count_images_per_class(labels_dir):
    # Count the number of instances per class
    ...
```

### 5. Visualizing Class Distribution

After preparing the dataset, it's helpful to visualize the class distribution. This can be done using pie and bar charts to show how many instances of each object class exist in the dataset.

```python
sns.set(style='whitegrid')
# Visualize class distribution with pie chart and bar chart
```

### 6. Prepare the YAML Configuration for YOLO Training

A YAML configuration file is created to define the paths for the training and validation images and labels. This file also specifies the number of object classes and their names, making it easier for the YOLO model to understand how to process the dataset.

```yaml
train: /kaggle/working/VOCdevkit/VOC2012/train/images
val: /kaggle/working/VOCdevkit/VOC2012/val/images

nc: 20  # number of classes
names: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
```

### 7. Train the YOLOv8 Model

The model is then trained using the prepared dataset. The training process uses the **YOLOv8** architecture, and transfer learning is applied to fine-tune the model for 25 epochs. The chosen image size and batch size help optimize the model’s performance.

```python
model = YOLO('yolov8n.pt') 
model.train(data=yaml_path, epochs=25, imgsz=256, batch=20)
```

### 8. Model Evaluation

Once training is complete, the model’s performance is evaluated using key metrics such as **Precision**, **Recall**, and **mAP (Mean Average Precision)**. These metrics provide insights into how well the model is detecting and classifying the objects in the validation dataset.

```python
# Model evaluation metrics (Precision, Recall, mAP)
metrics = { ... }
```

## Training Results

After completing the training, the model's performance was evaluated on the validation set with the following results:

- **Precision**: 0.77
- **Recall**: 0.55
- **mAP@50**: 0.63
- **mAP@50-95**: 0.45

These results indicate that the model is capable of detecting objects with high precision. While there's room for improvement, these metrics show that the model performs well for its current state.

## Directory Structure

The directory structure used to organize the dataset is as follows:

```
/VOCdevkit
    /VOC2012
        /train
            /images
            /labels
        /val
            /images
            /labels
        /annotations
        /ImageSets
            /Main
                train.txt
                val.txt
```

## Conclusion

This project walks you through the process of preparing a dataset, training a **YOLOv8 object detection model**, and evaluating its performance. The end-to-end solution includes everything from data preprocessing to model training, offering a solid foundation for object detection tasks.

While the model shows promising results, there are opportunities to further improve its performance with additional fine-tuning, more data, or different configurations. This approach can be adapted to other datasets, enabling real-time object detection in various applications.

---

Thanks for reading.
---
