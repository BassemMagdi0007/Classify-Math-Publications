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

- **Model Persistence**: Saves trained models and preprocessing artifacts (vocabulary, label encoders) for later reuse without retraining.


## Setup

### Repository Content
The repository contains the following files:
- `Example run/`: Folder containing example data run dependencies
- `classifier.py`: Main code file containing the model training and prediction logic
- `server_interaction.py`: Script for submitting predictions to server (Not Used)
- `create_training_data.py`: Creates training data by merging test data with ground truth labels <br> <br>
**_(Data run dependencies)_** <br>
- `model.keras`: Trained model
- `vocab.pickle`: Saved vocabulary for token conversion
- `label_encoder.pickle`: Saved label encoder
- `predictions.json`: Generated predictions for the test dataset
- `class_mapping.json`: JSON mapping of class IDs to category names

### Dependencies
- Python 3.8+
- TensorFlow 2.10+
- Keras 3.0+
- NumPy
- scikit-learn
- pickle _(included in Python standard library)_
- json _(included in Python standard library)_
- xml.etree.ElementTree _(included in Python standard library)_
- re _(included in Python standard library)_

#### Installation

```bash
pip install tensorflow numpy scikit-learn
```

### How to Run the Code
### **FIRST: Example Files:**
- Replace `path/to/example-test-data.jsonl` with the path to your example test data file.
- Replace `path/to/example-test-results.json` with the path to your example test results file.

#### 1) Create training data
```bash
python create_training_data.py path/to/example-test-data.jsonl path/to/example-test-results.json --output path/to/training_data.jsonl
```
> Output: `training_data.jsonl`
- Optionally, specify the `--output` argument to set the output file path. If not provided, it defaults to `training_data.jsonl`.

This will:
1. Loads data (example-test-data) and ground truth labels (example-test-results).
2. Merges data and labels into training data.
3. Saves training data to `training_data.jsonl` file.

#### 2) Training the Model
```bash
python classifier.py --train path/to/training_data.jsonl
```
> Outputs: `model.keras`, `vocab.pickle`, `label_encoder.pickle`

This will:
1. Load and process the training data
2. Train a neural network model (Data is splited into training/validation sets (80/20))
3. Save the model, vocabulary, and label encoder to disk

#### 3) Generating Test Predictions

```bash
python classifier.py --test path/to/example-test-data.jsonl
```
> Output: `predictions.json`
> 
This will:
1. Load the previously trained model and preprocessing objects
2. Process the test data
3. Generate and save the results in JSON format file `predictions.json`

#### 4) Evaluate the model 
```bash
python evaluate.py path/to/example-test-results.json path/to/predictions.json
```
>  <img width="366" alt="image" src="https://github.com/user-attachments/assets/3e54343f-c1cc-4ed7-9d11-606fc76373fe" />




### **THEN: Data Files:**
- Replace `path/to/training-data` with the path to your training data file.
- Replace `path/to/test-data.jsonl` with the path to your test data file.
  
#### 1) Training the Model
```bash
python classifier.py --train path/to/training-data.jsonl
```
>  <img width="425" alt="image" src="https://github.com/user-attachments/assets/9b0971ad-b41d-44e0-b2a9-3d795bb03c85" /> <br>
>  <img width="803" alt="image" src="https://github.com/user-attachments/assets/38dabee7-8afd-450f-b3fe-08b5ea87cc15" />

This will:
1. Train the neural network model on the training data
3. Save the model, vocabulary, and label encoder to disk

#### 2) Generating Test Predictions
```bash
python classifier.py --test path/to/test-data.jsonl
```
>  <img width="824" alt="image" src="https://github.com/user-attachments/assets/9d1faa0a-368a-4c2b-9d12-ad14bc4bc9e6" />

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
MAX_SEQ_LENGTH = 200  # Maximum length of input sequences
MIN_TOKEN_FREQ = 50   # Minimum frequency threshold for vocabulary inclusion
EMBEDDING_DIM = 256   # Dimension of embedding vectors
BATCH_SIZE = 64       # Number of samples per gradient update
EPOCHS = 20           # Maximum number of training epochs
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
- **Output**: Saves predictions to `predictions.json`.  

--- 


## Self-Evaluation and Design Decisions

### **Training Phase (Example Data)**  
The model was trained using the command:  
```bash  
python classifier.py --train data/training-data.jsonl  
```  
> <img width="470" alt="image" src="https://github.com/user-attachments/assets/da41d4af-5b2a-41c5-9f06-46e190b2fc2e" />

**Dataset Overview**:  
- **20,000 papers** with **759,837** total tokens.  
- After filtering rare tokens, the vocabulary reduced from **1,559 to 175 unique tokens**, achieving **98.99% coverage**.  
- Uncovered tokens (e.g., `'svg'`, `'tet'`, `'υ'`, `'when'`, `'on'`) were mapped to `<UNK>`.  

**Model Architecture**:  
- **Input vocabulary**: 175 tokens  
- **Categories**: 18 research fields  
- **Train/validation split**: 16,000 / 4,000 samples  
- **Sequence length**: Padded/truncated to 100 tokens  
- **Embeddings**: 128-dimensional  
- **Layers**: Two `Conv1D` layers for local feature extraction  
- **Regularization**: Dropout (0.3) in embedding and dense layers  
- **Early stopping**: Configured with `patience=3` (But not triggered; training completed all epochs "Design Decision").  

---

### **Training Progress**  
<img width="766" alt="image" src="https://github.com/user-attachments/assets/118188aa-d3ca-49fc-999e-4c4138099119" />
 

**Key Metrics**:  

| Epoch | Training Accuracy | Validation Accuracy | Validation Loss |
|-------|-------------------|---------------------|------------------|
| 1     | 21.2%             | 30.5%               | 2.2061           |
| 5     | 35.2%             | 40.5%               | 1.9501           |
| 10    | 49.8%             | 46.0%               | 1.7681           |
| 15    | 61.9%             | 48.3%               | 1.7208           |
| **16**| **62.6%**         | **51.3%**           | **1.7242**       |
| 17    | 68.9%             | 50.0%               | 1.7001           |
| 19    | 74.1%             | 49.8%               | 1.7892           |

> **Peak Validation Accuracy**: **51.3%** at **epoch 16**, with validation loss of **1.7242**.  
> This significantly surpasses the random baseline of ~5.6% (for 18 categories), demonstrating meaningful learning. However, **validation loss plateaued** and even slightly increased afterwards, suggesting mild **overfitting** in later epochs.

---

### **Model Artifacts**  
<img width="314" alt="image" src="https://github.com/user-attachments/assets/ffd9f54c-248a-45b4-a700-af654dcd4d48" />

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

<img width="665" alt="image" src="https://github.com/user-attachments/assets/6eb43b00-b43d-4ef5-88dc-139f9610afee" />
 
**Process Overview**:  
- Loaded pre-trained model and vocabulary.  
- Processed **2,000 test samples**, tokenizing and padding sequences to 100 tokens.  
- Predictions saved to `predictions.json` in the required format.  

**Example Tokenized Sequence**:  
```  
[2, 3, 4, 5, 3, 6, 7, 8, 3, 9, ...]  
```  
> This sequence represents structural and semantic tokens from a MathML formula, encoded using the trained vocabulary.

### **Example Data Evaluation**  
<img width="368" alt="image" src="https://github.com/user-attachments/assets/ff9601bd-a4cd-4a18-ac31-491583526936" />

---

## Output Format
 
 The results are stored as a JSON object, where the keys are the " Papers id"s and the values
 are the predicted classifications
 
<img width="228" alt="image" src="https://github.com/user-attachments/assets/b4771590-94f2-4711-8211-782de24f7036" />


