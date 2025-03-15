# Plant Disease Detection 

This Jupyter notebook contains a comprehensive implementation of a plant disease detection system using deep learning. The project leverages PyTorch to build, train, and evaluate a convolutional neural network that can identify various plant diseases from leaf images.

## Key Components of the Project

### 1. Data Acquisition and Preparation
- **Dataset**: The PlantVillage dataset is downloaded and extracted.
- **Custom Dataset Class**: A custom `PlantDiseaseDataset` class is implemented to handle the image data.
- **Data Augmentation**: Data augmentation techniques are applied using the Albumentations library.

### 2. Model Architecture
- **Dual-Output Neural Network**: A dual-output neural network based on ResNet50 is implemented:
  - One branch for disease classification.
  - Another branch for disease severity estimation.
- **Transfer Learning**: The model uses transfer learning by leveraging a pre-trained ResNet50 backbone.

### 3. Training Process
- **Dataset Splitting**: The dataset is split into training (80%) and validation (20%) sets.
- **Data Loaders**: Data loaders are configured with appropriate batch sizes and transformations.
- **Training**: The model is trained for 25 epochs using `CrossEntropyLoss` and Adam optimizer.
- **Learning Rate Scheduling**: Learning rate scheduling is implemented to improve convergence.
- **Metrics Tracking**: Training metrics (loss and accuracy) are tracked and visualized.

### 4. Inference and Visualization
- **Prediction Functions**: Functions for making predictions on individual images are implemented.
- **Visualization Tools**: Visualization tools to display the original image alongside predictions.
- **Activation Map Visualization**: Activation map visualization to highlight the regions the model focuses on.

### 5. Results
- **High Accuracy**: The model achieves high accuracy (>97%) on the validation set.
- **Training Convergence**: Training converges well, with both training and validation losses decreasing steadily.
- **Disease Identification**: The model successfully identifies different plant diseases from leaf images.

## Summary

This project demonstrates a complete machine learning pipeline for plant disease detection. It starts with data preparation, implements a sophisticated deep learning model with dual outputs (disease classification and severity estimation), trains the model effectively, and provides tools for inference and visualization.

The implementation takes advantage of modern deep learning practices like transfer learning, data augmentation, and learning rate scheduling. The notebook is well-structured and includes visualization of training metrics to monitor the learning process.

The high accuracy achieved by the model suggests that it could be a valuable tool for early detection of plant diseases, potentially helping farmers take timely action to prevent crop losses. The dual-output nature of the model provides not just disease identification but also an estimation of severity, which adds practical value to the system.
