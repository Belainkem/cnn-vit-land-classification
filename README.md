# AI Capstone Project - Land Classification

An AI capstone project that compares different deep learning approaches for land classification, specifically distinguishing between agricultural and non-agricultural areas using satellite imagery.

## Project Overview

This project implements and compares various deep learning models for binary land classification:
- **Agricultural land** (class 1)
- **Non-agricultural land** (class 0)

The project explores multiple frameworks and architectures:
- Keras-based CNN models
- PyTorch-based CNN models
- Vision Transformer (ViT) implementations
- Data augmentation techniques
- Comparative analysis of model performance

## Project Structure

```
AI_Capstone_Project/
├── notebooks/              # Jupyter notebooks for different experiments
│   ├── 01_memory_vs_generator.ipynb
│   ├── 02_keras_augmentation.ipynb
│   ├── 03_pytorch_augmentation.ipynb
│   ├── 04_keras_classifier.ipynb
│   ├── 05_pytorch_classifier.ipynb
│   ├── 06_comparative_analysis.ipynb
│   ├── 07_keras_vit.ipynb
│   ├── 08_pytorch_vit.ipynb
│   └── 09_land_classification_evaluation.ipynb
├── images_dataSAT/         # Dataset directory
│   ├── class_0_non_agri/   # Non-agricultural images
│   └── class_1_agri/       # Agricultural images
├── models/                 # Trained model files
│   └── pytorch_cnn_vit_ai_capstone_model_state_dict.pth
└── README.md
```

## Notebooks Description

1. **01_memory_vs_generator.ipynb** - Comparison of memory-based vs generator-based data loading
2. **02_keras_augmentation.ipynb** - Data augmentation techniques using Keras
3. **03_pytorch_augmentation.ipynb** - Data augmentation techniques using PyTorch
4. **04_keras_classifier.ipynb** - Keras CNN classifier implementation
5. **05_pytorch_classifier.ipynb** - PyTorch CNN classifier implementation
6. **06_comparative_analysis.ipynb** - Comparative analysis of Keras and PyTorch models
7. **07_keras_vit.ipynb** - Vision Transformer implementation using Keras
8. **08_pytorch_vit.ipynb** - Vision Transformer implementation using PyTorch
9. **09_land_classification_evaluation.ipynb** - Final evaluation and results

## Technologies Used

- **Python** - Main programming language
- **TensorFlow/Keras** - Deep learning framework
- **PyTorch** - Deep learning framework
- **Jupyter Notebooks** - Interactive development environment
- **NumPy, Matplotlib** - Data manipulation and visualization
- **scikit-learn** - Metrics and evaluation

## Model Architectures

### CNN Models
- 4 Convolutional layers with increasing filters (32, 64, 128, 256)
- 5 Dense layers (4 hidden + 1 output)
- MaxPooling for dimensionality reduction
- Softmax activation for binary classification

### Vision Transformer (ViT)
- Transformer-based architecture for image classification
- Self-attention mechanisms
- Patch-based image processing

## Dataset

The dataset contains satellite images of:
- **Class 0 (Non-agricultural)**: Urban areas, cities, buildings
- **Class 1 (Agricultural)**: Farmlands, agricultural fields

## Results

The project includes comparative analysis of:
- Model accuracy
- Precision and recall metrics
- F1-scores
- Confusion matrices
- Training time and efficiency

## Usage

1. Clone the repository:
```bash
git clone https://github.com/Belainkem/AI_Capstone_Project.git
cd AI_Capstone_Project
```

2. Install required dependencies:
```bash
pip install tensorflow torch torchvision numpy matplotlib scikit-learn jupyter
```

3. Run the notebooks in order to reproduce the experiments

## Author

Aditya Pandharkar (Belainkem)

## License

This project is part of an AI Capstone course and is for educational purposes.

