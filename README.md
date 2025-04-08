Classify Math Publications
==========================

## Table of Contents

- [Introduction](#introduction)
  - [Key Features](#key-features)
- [Setup](#setup)
  - [Repository Content](#repository-content)
  - [Dependencies](#dependencies)
  - [How to Run the Code](#how-to-run-the-code)
  - [Used Libraries](#used-libraries)
- [Code Structure](#code-structure)
- [Self-Evaluation and Design Decisions](#self-evaluation-and-design-decisions)
- [Output Format](#output-format)

# Introduction

This project implements a neural network-based classifier for mathematical publications using the formulas they contain as distinguishing features. The system processes arXiv publications converted to HTML with LaTeXML, where mathematical formulas are represented in MathML format. The classifier analyzes these formulas to predict the academic category of each paper.

### Key Features

- **Robust MathML Processing**: Implements sophisticated tokenization of MathML formulas that handles XML structure, normalizes numerical values and operators, and gracefully manages parsing errors.

- **Intelligent Vocabulary Construction**: Builds a frequency-based vocabulary that filters rare tokens to reduce dimensionality while maintaining good coverage of the input space.

- **Neural Network Architecture**: Employs a convolutional neural network (CNN) with embedding layers specifically designed for processing sequences of mathematical notation tokens.

- **Comprehensive Training Pipeline**: Includes data loading, preprocessing, model training with early stopping, and evaluation components in a reproducible workflow.

- **Production-Ready Deployment**: Provides functions for both batch processing of test files and API-style individual paper classification.

- **Model Persistence**: Saves trained models and preprocessing artifacts (vocabulary, label encoders) for later reuse without retraining.


## Setup

### Repository Content
The repository contains the following files:
- `classifier.py`: Main code file containing the model training and prediction logic
- `server_interaction.py`: Script provided by the assignment for submitting predictions
- `model.keras`: Trained model
- `vocab.pickle`: Saved vocabulary for token conversion
- `label_encoder.pickle`: Saved label encoder 
- `test_results.json`: Generated predictions for the test dataset
- `solution_summary.md`: Summary of the solution approach
- `README.md`: This file

### Dependencies

- Python 3.8+
- TensorFlow 2.10+
- Keras 3.0+
- NumPy
- scikit-learn
- pickle (included in Python standard library)
- json (included in Python standard library)
- xml.etree.ElementTree (included in Python standard library)
- re (included in Python standard library)

#### Installation

```bash
pip install tensorflow numpy scikit-learn
```

### How to Run the Code
### **FIRST: Example Files:**

#### 1) Create training data
```bash
python create_training_data.py path/to/example-test-data.jsonl path/to/example-test-results.json --output path/to/training_data.jsonl
```
- Replace `path/to/example-test-data.jsonl` with the path to your test data file.
- Replace `path/to/example-test-results.json` with the path to your test results file.
- Optionally, specify the `--output` argument to set the output file path. If not provided, it defaults to `training_data.jsonl`.

This will:
1. Loads test data and ground truth labels.
2. Merges data and labels into training data.
3. Saves training data to `training_data.jsonl` file.

#### 2) Training the Model
```bash
python classifier.py --train path/to/training_data.jsonl
```

This will:
1. Load and process the training data
2. Train a neural network model
3. Save the model, vocabulary, and label encoder to disk

#### 3) Generating Test Predictions

```bash
python classifier.py --test path/to/example-test-data.jsonl
```

This will:
1. Load the previously trained model and preprocessing objects
2. Process the test data
3. Generate and save the results in JSON format file `predictions.json`

#### 4) Evaluate the model 
```bash
python evaluate.py path/to/example-test-results.json path/to/predictions.json
```

### **THEN: Data Files:**
#### 1) Training the Model
```bash
python classifier.py --train path/to/training-data.jsonl
```
This will:
1. Train the neural network model on the training data
3. Save the model, vocabulary, and label encoder to disk

#### 2) Generating Test Predictions
```bash
python classifier.py --test path/to/test-data.jsonl
```

This will:
1. Load the previously trained model and preprocessing objects
2. Process the test data
3. Generate and save the results in JSON format file `predictions.json`



### Used Libraries

The implementation leverages several Python libraries to create an efficient and maintainable solution:

- **Core Libraries**:
  - `xml.etree.ElementTree`: For parsing MathML XML structure
  - `re`: Regular expressions for text processing and normalization
  - `json`: Handling input/output of JSON data files
  - `collections.Counter`: Vocabulary construction and token counting

- **Machine Learning Stack**:
  - `numpy`: Numerical operations and array handling
  - `scikit-learn` (`LabelEncoder`, `train_test_split`): Data preprocessing and evaluation utilities
  - `keras`: High-level neural network API for model construction and training
  - `tensorflow` (as Keras backend): Efficient tensor operations and GPU acceleration

- **Utility Libraries**:
  - `pickle`: Serialization of model artifacts for persistence
  - `argparse`: Command-line interface for training and prediction scripts


## Code Structure

### The implementation Skeleton:

1. **Data Preprocessing & Feature Engineering**:
   - `tokenize_mathml()`: Converts MathML formulas to normalized tokens
   - `build_vocabulary()`: Constructs frequency-based token vocabulary
   - `load_and_process_data()`: Complete pipeline from raw data to training-ready sequences

2. **Model Creation & Training**:
   - `create_model()`: Defines CNN architecture with embedding and classification layers
   - `train_and_save_model()`: End-to-end training with early stopping and model persistence

3. **Test Data Processing & Prediction**:
   - `process_test_data()`: Applies preprocessing to test data and generates predictions
   - `get_classifications()`: API-style function for real-time classification

4. **Main Execution**:
   - Command-line interface supporting both training and prediction modes
   - Handles model loading and saving for efficient reuse

### Code:

### **1. Imports and Configuration**
```python
import xml.etree.ElementTree as ET
import re
import json
import numpy as np
import os
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Keras backend configuration
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.callbacks import EarlyStopping
```
- **Core Libraries**:  
  - `xml.etree.ElementTree`: For parsing MathML formulas.  
  - `re`: Regular expressions for text normalization.  
  - `json`: Handling JSON-formatted paper data.  
  - `numpy`: Numerical operations and array management.  
- **ML Tools**:  
  - `LabelEncoder`: Converts class labels to numerical indices.  
  - `train_test_split`: Creates training/validation splits.  
- **Deep Learning**:  
  - Keras with TensorFlow backend for model construction and training.  

### **2. Hyperparameters**
```python
MAX_SEQ_LENGTH = 100   # Maximum input sequence length  
MIN_TOKEN_FREQ = 10    # Minimum token frequency for vocabulary  
EMBEDDING_DIM = 128    # Dimension of embedding vectors  
BATCH_SIZE = 32        # Training batch size  
EPOCHS = 15            # Maximum training epochs  
```
- **Sequence Handling**: `MAX_SEQ_LENGTH` truncates/pads token sequences.  
- **Vocabulary Filtering**: `MIN_TOKEN_FREQ` prunes rare tokens to reduce noise.  
- **Model Architecture**: `EMBEDDING_DIM` defines the dense representation space.  

---

### **3. Data Processing Pipeline**
#### **`tokenize_mathml(formula)`**
```python
def tokenize_mathml(formula):
    # Parses MathML into normalized tokens (e.g., numbers → `<NUMBER>`).
    # Handles XML namespaces, mathematical operators, and error cases.
```
- **Input**: Raw MathML string.  
- **Output**: List of normalized tokens (e.g., `['mi', '<NUMBER>', '@variable']`).  
- **Key Steps**:  
  1. XML namespace stripping.  
  2. Number/operator normalization.  
  3. Attribute handling (e.g., `@class` annotations).  

#### **`build_vocabulary(token_lists)`**
```python
def build_vocabulary(token_lists):
    # Constructs frequency-based vocabulary with rare-token filtering.
```
- **Input**: List of tokenized papers.  
- **Output**: Dictionary mapping tokens to indices (e.g., `{'<PAD>': 0, '<UNK>': 1, ...}`).  
- **Filtering**: Tokens with frequency < `MIN_TOKEN_FREQ` are mapped to `<UNK>`.  

#### **`load_and_process_data(file_path)`**
```python
def load_and_process_data(file_path):
    # End-to-end pipeline: JSONL → tokenized sequences → train/val splits.
```
- **Steps**:  
  1. Loads JSONL papers and extracts formulas.  
  2. Tokenizes formulas and builds vocabulary.  
  3. Converts tokens to padded integer sequences.  
  4. Splits data into training/validation sets (80/20).  

---

### **4. Model Architecture**
#### **`create_model(vocab_size, num_categories)`**
```python
def create_model(vocab_size, num_categories):
    # Builds CNN-based classifier with embeddings and dropout.
```
- **Layers**:  
  1. **Embedding**: Maps token indices to dense vectors (`EMBEDDING_DIM=128`).  
  2. **Convolutional**: Two 1D conv layers (kernel size=5) with ReLU activation.  
  3. **Pooling**: Max-pooling followed by global pooling.  
  4. **Dense**: Fully connected layers with dropout (`rate=0.3`).  
- **Output**: Softmax over paper categories.  

#### **Training Configuration**
- **Optimizer**: Adam.  
- **Loss**: Sparse categorical cross-entropy.  
- **Early Stopping**: Monitors `val_accuracy` with `patience=3`.  

---

### **5. Training & Evaluation**
#### **`train_and_save_model()`**
```python
def train_and_save_model(training_file, model_output_path='model.keras'):
    # Orchestrates data loading, training, and model persistence.
```
- **Saved Artifacts**:  
  - Trained Keras model (`model.keras`).  
  - Vocabulary and label encoder (pickle files).  
  - Class mapping (JSON).  
- **Validation**: Reports accuracy on held-out data.  

---

### **6. Inference Pipeline**
#### **`process_test_data()`**
```python
def process_test_data(test_file, vocab, label_encoder, model):
    # Processes test papers and generates predictions.
```
- **Steps**:  
  1. Tokenizes test formulas using the trained vocabulary.  
  2. Pads/truncates sequences to `MAX_SEQ_LENGTH`.  
  3. Runs model inference and decodes predictions.  

#### **`get_classifications(request)`**
```python
def get_classifications(request):
    # API-compatible function for real-time predictions.
```
- **Input**: List of paper dictionaries (from server requests).  
- **Output**: Predicted categories.  

---

### **7. Command-Line Interface**
```python
def main():
    # Supports --train (training mode) and --test (inference mode).
```
- **Usage**:  
  ```bash
  python classifier.py --train papers_train.jsonl --test papers_test.jsonl
  ```
- **Output**: Saves predictions to `predictions.json`.  

--- 


## Self-Evaluation and Design Decisions

### **Training Phase**  
The model was trained using the command:  
```bash  
python classifier.py --train data/training-data.jsonl  
```  

<img width="709" alt="image" src="https://github.com/user-attachments/assets/3ddf12cf-37d3-4900-aacf-81a4de76a08f" />  

**Dataset Overview**:  
- **20,000 papers** with **7.6 million tokens** total.  
- After filtering rare tokens (minimum frequency=10), the vocabulary reduced from **6,407 to 1,196 unique tokens**, achieving **99.85% coverage**.  
- Uncovered tokens (e.g., `'mu'`, `'caf'`, `'𝗸𝗽𝗰'`) were mapped to `<UNK>`.  

**Model Architecture**:  
- **Input vocabulary**: 1,196 tokens  
- **Categories**: 18 research fields  
- **Train/validation split**: 16,000 / 4,000 samples  
- **Sequence length**: Padded/truncated to 100 tokens  
- **Embeddings**: 128-dimensional  
- **Layers**: Two `Conv1D` layers for local feature extraction  
- **Regularization**: Dropout (0.3) in embedding and dense layers  
- **Early stopping**: Configured with patience=3 (not triggered; training completed all epochs).  

---

### **Training Progress**  
<img width="900" alt="image" src="https://github.com/user-attachments/assets/4a4af77d-ccfe-4872-8cf1-c0ede84a794b" />  

**Key Metrics**:  

| Epoch | Training Accuracy | Validation Accuracy | Validation Loss |  
|-------|-------------------|---------------------|------------------|  
| 1     | 33.1%             | 45.0%               | 1.7840           |  
| 5     | 59.9%             | 56.1%               | 1.4703           |  
| 10    | 67.0%             | 57.4%               | 1.4546           |  
| **13**| **69.7%**         | **57.6%**           | **1.5082**       |  
| 15    | 72.2%             | 56.4%               | 1.5733           |  

> **Peak Validation Accuracy**: **57.6%** at epoch 13, with validation loss of 1.5082. This exceeds the baseline of 25%, demonstrating the model’s ability to generalize to unseen data despite moderate overfitting in later epochs.  

---

### **Model Artifacts**  
<img width="340" alt="image" src="https://github.com/user-attachments/assets/0bdf598a-aecc-4f86-b58b-09950449454b" />  

The trained model and supporting files were saved as:  
- `model.keras` (architecture and weights)  
- `vocab.pickle` (token-to-index mappings)  
- `label_encoder.pickle` (category label encoder)  

---

### **Testing and Predictions**  
Predictions were generated with:  
```bash  
python classifier.py --test data/test-data.jsonl  
```  

<img width="1262" alt="image" src="https://github.com/user-attachments/assets/3eead180-86b3-4fc3-87a9-610b4036a059" />  

**Process Overview**:  
- Loaded pre-trained model and vocabulary.  
- Processed **2,000 test samples**, tokenizing and padding sequences to 100 tokens.  
- Predictions saved to `predictions.json` in the required format.  

**Example Tokenized Sequence**:  
```  
[2, 3, 78, 5, 6, 2, 16, 5, 49, 3, 44, 16, ...]  
```  
> This sequence represents structural and semantic tokens from a MathML formula, encoded using the trained vocabulary.  

---

