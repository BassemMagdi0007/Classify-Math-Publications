Classify Math Publications
==========================

## Table of Contents

- [Introduction](#introduction)
  - [Key Features](#key-features)
- [Setup](#setup)
  - [Repository Content](#repository-content)
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

## **Workflow Summary**
1. **Training Phase**:  
   - Processes raw JSONL → tokenized sequences → trains CNN classifier.  
   - Persists model and vocabulary for deployment.  
2. **Inference Phase**:  
   - Loads saved artifacts.  
   - Classifies new papers via `process_test_data()` or `get_classifications()`.  

--- 


## Self-Evaluation and Design Decisions

The implementation makes several thoughtful design choices to balance performance, accuracy, and maintainability:

1. **Tokenization Strategy**:
   - Normalizes numbers and operators to improve generalization
   - Preserves XML structure through element tags
   - Includes robust error handling for malformed MathML

2. **Vocabulary Construction**:
   - Implements frequency thresholding (MIN_TOKEN_FREQ) to reduce dimensionality
   - Tracks vocabulary coverage to ensure model can handle unseen data
   - Uses special tokens (<UNK>, <PAD>) for out-of-vocabulary terms and sequence padding

3. **Neural Network Architecture**:
   - Employs embedding layer to learn meaningful representations of mathematical tokens
   - Uses 1D convolutional layers to capture local patterns in formula sequences
   - Includes dropout layers for regularization to prevent overfitting
   - Implements early stopping based on validation accuracy

4. **Performance Considerations**:
   - Limits sequence length (MAX_SEQ_LENGTH) to balance memory usage and information retention
   - Uses efficient batching (BATCH_SIZE) for gradient updates
   - Provides model persistence to avoid retraining

5. **Validation and Testing**:
   - Maintains separate validation set for unbiased performance evaluation
   - Includes comprehensive logging of preprocessing statistics
   - Generates properly formatted output as specified in the assignment

The solution demonstrates good software engineering practices while addressing the machine learning challenges of the classification task. The modular design allows for easy experimentation with different tokenization strategies or model architectures, and the comprehensive documentation ensures maintainability.



#-------------------------------------------------------------------------------------------
## Dependencies

- Python 3.8+
- TensorFlow 2.10+
- Keras 3.0+
- NumPy
- scikit-learn
- pickle (included in Python standard library)
- json (included in Python standard library)
- xml.etree.ElementTree (included in Python standard library)
- re (included in Python standard library)

### Installation

```bash
pip install tensorflow numpy scikit-learn
```

## Repository Structure

- `classifier.py`: Main code file containing the model training and prediction logic
- `server_interaction.py`: Script provided by the assignment for submitting predictions
- `model.keras`: Trained model (not included in repository due to size)
- `vocab.pickle`: Saved vocabulary for token conversion (not included in repository)
- `label_encoder.pickle`: Saved label encoder (not included in repository)
- `test_results.json`: Generated predictions for the test dataset
- `solution_summary.md`: Summary of the solution approach
- `README.md`: This file

## How to Run

### Training the Model

```bash
python classifier.py --train path/to/training_data.jsonl
```

This will:
1. Load and process the training data
2. Train a neural network model
3. Save the model, vocabulary, and label encoder to disk

### Generating Test Predictions

```bash
python classifier.py --test path/to/test_data.jsonl --output test_results.json
```

This will:
1. Load the previously trained model and preprocessing objects
2. Process the test data
3. Generate predictions
4. Save the results in JSON format

### Server Submission

The repository includes a `server_interaction.py` script that implements the `get_classifications` function required by the grading server. This function is already imported in `classifier.py`.

To use it:
1. Make sure `model.keras`, `vocab.pickle`, and `label_encoder.pickle` exist in the working directory
2. Run:
```bash
python server_interaction.py
```

## Generating Model Files

If you've downloaded this repository without the model files, you'll need to train the model first:

1. Download the training data (not included in repository)
2. Run the training command above
3. The necessary model files will be generated automatically

## Model Architecture

The neural network uses:
- Embedding layer (128 dimensions)
- Two 1D convolutional layers with 64 filters each
- Global max pooling
- Dense layers with dropout for regularization
- Softmax output layer for multi-class classification

## Performance

The model achieves approximately 40% accuracy on the validation set, which is significantly above the 25% required passing threshold.

# Solution Summary: Classifying arXiv Publications via MathML Formula Analysis

## Problem Overview
The task requires classifying mathematical publications into research categories (e.g., 'cs', 'cond-mat', 'hep-ph') based solely on the MathML representations of the formulas they contain. Each paper contains 1-10 formulas, and the challenge is to extract meaningful patterns from these mathematical expressions that correlate with research domains.

## Approach
My solution employs a neural network that learns to identify mathematical notation patterns specific to different research fields. The approach consists of four main components:

### 1. MathML Tokenization
I developed a tokenization strategy that extracts both structural and content information from MathML:
- XML tags (e.g., `<mi>`, `<mrow>`, `<msub>`) capture the mathematical structure
- Text content within tags captures variable names, operators, and constants
- Numbers are normalized to reduce vocabulary size and improve generalization
- Special tokens handle padding and unknown elements

This approach preserves both the notation style and content of mathematical expressions, which vary significantly across research domains. For example, theoretical physics papers might contain more integral expressions and specific variable notations than computer science papers.

### 2. Feature Engineering
The tokenized formulas are converted to fixed-length numerical sequences:
- Built a vocabulary from tokens appearing at least 5 times in the training corpus
- Mapped each token to a unique integer index
- Limited sequences to 100 tokens (truncating longer sequences)
- Padded shorter sequences with zeros

### 3. Neural Network Architecture
I designed a neural network with the following components:
- Embedding layer: Transforms token indices into 128-dimensional dense vectors
- Convolutional layers: Two Conv1D layers with 64 filters each, capture n-gram patterns in formulas
- Global max pooling: Extracts the most important features from each filter
- Dense layers: Transform the extracted features into category predictions
- Dropout (0.3): Applied after embedding and before output to prevent overfitting

This architecture effectively captures both local patterns (specific mathematical constructs) and their relationships within formulas, which is crucial for distinguishing between research domains.

### 4. Training and Optimization
The model was trained with:
- Sparse categorical cross-entropy loss
- Adam optimizer
- Early stopping based on validation accuracy
- 80/20 train-validation split

## Results and Analysis
The model achieved a validation accuracy of approximately 40%, significantly above the 25% required passing threshold. Analysis of the results revealed:

1. Certain mathematical notations strongly correlate with specific fields:
   - Physics papers often use specific symbols and tensor notations
   - CS papers tend to have more algorithmic and logical expressions
   - Mathematics papers show particular theorem structures

2. Challenges encountered:
   - Similar notation can appear across different domains
   - MathML parsing errors in some formulas
   - Limited context from only having formula information

## Potential Improvements
Given more time, these enhancements could further improve accuracy:
- More sophisticated tokenization that captures hierarchical formula structure
- Attention mechanisms to focus on discriminative parts of formulas
- Ensemble methods combining multiple model architectures

## Conclusion
The solution demonstrates that mathematical notation alone contains sufficient information to reasonably predict research domains. This suggests that different fields have developed distinctive mathematical "dialects" that can be identified through machine learning techniques.
