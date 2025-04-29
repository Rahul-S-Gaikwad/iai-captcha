# CAPTCHA-breaking in Darknet Marketplaces

## Project Description
This repository contains the implementation for the final project in the Introduction to Artificial Intelligence course (95-891) at Carnegie 
Mellon University. This project is focused on developing and training models for CAPTCHA recognition, with components for text and 
object detection. The system includes preprocessing, training, and testing pipelines for both text-based and object-based CAPTCHAs.

## File Structure
```
merged/
├── config.py                 # Global configuration parameters
├── data/                     # Data directories
│   ├── raw_training_data/    # Raw training data
│   ├── training_data/        # Processed training data
│   ├── validation_data/      # Validation data
│   ├── test_data/            # Test data
│   └── scraped_captchas/     # Scraped CAPTCHA samples
├── models/                   # Trained models
│   ├── object/               # Object detection models
│   └── text/                 # Text recognition models
├── preprocessing/            # Preprocessing modules
│   ├── object/               # Object detection preprocessing
│   └── text/                 # Text recognition preprocessing
├── training/                 # Training modules
│   ├── trainer.py            # Main training script
│   ├── prepare_training.py   # Training data preparation
│   └── network.py            # Neural network architecture
└── testing/                  # Testing and evaluation modules
    ├── object/               # Object detection testing
    └── text/                 # Text recognition testing
```

## Configuration
The project uses a centralized configuration file (`config.py`) that sets global parameters:
- EPOCHS: Number of training epochs
- BATCH_SIZE: Training batch size
- SPLIT_RATE: Train/validation split ratio
- Various directory paths for data and models

## Usage Instructions

### Prerequisites
- Python 3.9
- Required Python packages listed in `requirements.txt`

### Data Preparation
1. Place raw CAPTCHA data in the `data/raw_training_data` directory
2. Run preprocessing scripts to prepare the data: 
   ```bash
   python preprocessing/object/create_training_data.py  # For object CAPTCHAs
   python preprocessing/text/create_training_data.py  # For text CAPTCHAs
   ```

### Training
1. Train the models:
   ```bash
   python training/trainer.py
   ```
2. Models are saved in the `models` directory

### Testing
1. Place test data in the `data/test_data` directory
2. Run the testing scripts:
   ```bash
   python testing/object/tester.py  # For object CAPTCHAs
   python testing/text/tester.py  # For text CAPTCHAs
   ```
