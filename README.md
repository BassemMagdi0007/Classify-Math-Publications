Classify Math Publications
==========================


# arXiv Math Publication Classifier

This repository contains a solution for classifying arXiv mathematical publications into research categories based on the MathML representations of the formulas they contain.

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
