# Import necessary libraries for data processing, ML, and file handling
import xml.etree.ElementTree as ET
import re
import json
import numpy as np
import os
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
#-------------------------------------------------------------------------------------------
# Configure Keras to use TensorFlow backend
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.callbacks import EarlyStopping
#-------------------------------------------------------------------------------------------
# Model hyperparameters and configuration settings
MAX_SEQ_LENGTH = 200  # Maximum length of input sequences
MIN_TOKEN_FREQ = 50   # Minimum frequency threshold for vocabulary inclusion
EMBEDDING_DIM = 256   # Dimension of embedding vectors
BATCH_SIZE = 64       # Number of samples per gradient update
EPOCHS = 20           # Maximum number of training epochs
#-------------------------------------------------------------------------------------------
"""1. Data Preprocessing & Feature Engineering"""
#-------------------------------------------------

def tokenize_mathml(formula):
    """
    Parse MathML formula into tokens with normalization for better generalization.
    Handles XML structure, normalizes numbers and operators, and manages errors.
    """
    try:
        # Remove XML namespace prefixes if present
        formula = re.sub(r'<(\w+):', '<', formula)
        formula = re.sub(r'</(\w+):', '</', formula)
        
        root = ET.fromstring(formula)
        tokens = []
        
        for elem in root.iter():
            if elem == root:
                continue  # Skip root element
                
            # Add element tag
            tokens.append(elem.tag)
            
            # Process text content
            if elem.text and elem.text.strip():
                text = elem.text.strip().lower()  # Normalize case
                
                # Normalize numbers
                if re.match(r'^-?\d+\.?\d*([eE][+-]?\d+)?$', text):
                    tokens.append('<NUMBER>')
                # Normalize common symbols
                elif text in ['+', '-', '*', '/', '=', '<', '>']:
                    tokens.append('<OPERATOR>')
                else:
                    tokens.append(text)
                    
            # Handle attributes
            if 'class' in elem.attrib:
                tokens.append(f"@{elem.attrib['class']}")
                
        return tokens
        
    except Exception as e:
        print(f"Error parsing formula: {str(e)}")
        return ['<ERROR>']

def build_vocabulary(token_lists):
    """
    Create frequency-based vocabulary mapping from token lists.
    Filters out rare tokens to reduce dimensionality and prevent overfitting.
    """
    # Count all tokens
    token_counts = Counter()
    for tokens in token_lists:
        token_counts.update(tokens)
    
    print(f"Found {len(token_counts)} unique tokens before filtering")
    
    # Create vocabulary with frequent tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    current_idx = 2
    
    for token, count in token_counts.items():
        if count >= MIN_TOKEN_FREQ and token not in vocab:
            vocab[token] = current_idx
            current_idx += 1
    
    print(f"Vocabulary size after filtering: {len(vocab)}")
    print(f"Max token index: {max(vocab.values())}")
    
    return vocab

def validate_vocabulary_coverage(token_lists, vocab):
    """
    Analyze vocabulary coverage statistics to ensure model can handle input data.
    Reports percentage of tokens covered and provides examples of uncovered tokens.
    """
    total_tokens = 0
    covered_tokens = 0
    
    for tokens in token_lists:
        for token in tokens:
            total_tokens += 1
            if token in vocab:
                covered_tokens += 1
    
    coverage = (covered_tokens / total_tokens) * 100
    print(f"Vocabulary coverage: {coverage:.2f}%")
    
    # Show sample uncovered tokens
    uncovered = set()
    for tokens in token_lists:
        for token in tokens:
            if token not in vocab:
                uncovered.add(token)
                if len(uncovered) >= 5:
                    break
    
    if uncovered:
        print("Sample uncovered tokens:", list(uncovered)[:5])

def load_and_process_data(file_path):
    """
    Main data pipeline: loads JSONL data, extracts tokens, builds vocabulary,
    converts to sequences, and prepares train/validation splits for model training.
    """
    papers = []
    categories = []
    all_tokens = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Process formulas for this paper
            paper_tokens = []
            for formula in data['formulas']:
                tokens = tokenize_mathml(formula)
                paper_tokens.extend(tokens)
                all_tokens.extend(tokens)
            
            papers.append(paper_tokens)
            categories.append(data['classification'])
    
    print(f"Loaded {len(papers)} papers with {len(all_tokens)} total tokens")
    
    # Build vocabulary
    vocab = build_vocabulary(papers)
    validate_vocabulary_coverage(papers, vocab)
    
    # Convert tokens to sequences
    sequences = []
    for paper_tokens in papers:
        # Convert tokens to indices with bounds checking
        sequence = []
        for token in paper_tokens:
            idx = vocab.get(token, vocab['<UNK>'])
            if idx >= len(vocab):  # Extra safety check
                idx = vocab['<UNK>']
            sequence.append(idx)
        
        # Truncate or pad sequence
        if len(sequence) > MAX_SEQ_LENGTH:
            sequence = sequence[:MAX_SEQ_LENGTH]
        else:
            sequence = sequence + [vocab['<PAD>']] * (MAX_SEQ_LENGTH - len(sequence))
        
        sequences.append(sequence)
    
    # Convert categories to indices
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(categories)
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(y)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val, vocab, label_encoder
#-------------------------------------------------------------------------------------------
"""2) Model Creation & Training"""
#----------------------------------

def create_model(vocab_size, num_categories):
    model = keras.models.Sequential([
        keras.layers.Input(shape=(MAX_SEQ_LENGTH,)),
        keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            mask_zero=True
        ),
        keras.layers.Dropout(0.5),  # Increased regularization
        keras.layers.Conv1D(128, 5, activation='relu'),  # More filters
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(128, 5, activation='relu'),  # More filters
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(256, activation='relu'),  # Larger dense layer
        keras.layers.Dropout(0.5),  # Increased regularization
        keras.layers.Dense(num_categories, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_save_model(training_file, model_output_path='model.keras', 
                        vocab_output_path='vocab.pickle', 
                        encoder_output_path='label_encoder.pickle'):
    """
    End-to-end training function: processes data, builds model, trains with early stopping,
    evaluates performance, and persists model artifacts for later use.
    """
    # Load and process data
    X_train, X_val, y_train, y_val, vocab, label_encoder = load_and_process_data(training_file)
    
    # Create and train model
    vocab_size = len(vocab)
    num_categories = len(label_encoder.classes_)
    
    print(f"\nTraining model with:")
    print(f"- Vocabulary size: {vocab_size}")
    print(f"- Number of categories: {num_categories}")
    print(f"- Training samples: {len(X_train)}")
    print(f"- Validation samples: {len(X_val)}")
    
    model = create_model(vocab_size, num_categories)
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate on validation set
    print("\nEvaluation results:")
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Save model and preprocessing objects
    model.save(model_output_path)
    with open(vocab_output_path, 'wb') as f:
        pickle.dump(vocab, f)
    with open(encoder_output_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save class mapping
    class_mapping = {i: cls for i, cls in enumerate(label_encoder.classes_)}
    with open('class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    print(f"\nSaved model to {model_output_path}")
    print(f"Saved vocabulary to {vocab_output_path}")
    print(f"Saved label encoder to {encoder_output_path}")
    
    return model, vocab, label_encoder
#-------------------------------------------------------------------------------------------
"""3. Test Data Processing & Prediction"""
#-----------------------------------------

def process_test_data(test_file, vocab, label_encoder, model):
    """
    Process test dataset and generate predictions using trained model.
    Handles data loading, tokenization, sequence conversion, and result formatting.
    """
    # Read test data
    test_papers = []
    paper_ids = []
    
    with open(test_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            paper_ids.append(data['id'])
            
            # Process formulas
            paper_tokens = []
            for formula in data['formulas']:
                tokens = tokenize_mathml(formula)
                paper_tokens.extend(tokens)
            
            test_papers.append(paper_tokens)
    
    # Convert to sequences
    sequences = []
    for paper_tokens in test_papers:
        sequence = [vocab.get(token, vocab['<UNK>']) for token in paper_tokens]
        
        # Truncate or pad sequence
        if len(sequence) > MAX_SEQ_LENGTH:
            sequence = sequence[:MAX_SEQ_LENGTH]
        else:
            sequence = sequence + [vocab['<PAD>']] * (MAX_SEQ_LENGTH - len(sequence))
        
        sequences.append(sequence)
    
    # Make predictions
    X_test = np.array(sequences)
    predictions = model.predict(X_test)
    pred_indices = np.argmax(predictions, axis=1)
    pred_categories = label_encoder.inverse_transform(pred_indices)

    print(f"Processed {len(sequences)} test samples")
    print(f"Sample processed sequence: {sequences[0]}")

    
    # Create dictionary of predictions
    results = {paper_id: category for paper_id, category in zip(paper_ids, pred_categories)}
    
    # # Save predictions in JSON format with pretty-printing (new line for each entry)
    # with open('my_predictions.json', 'w') as f:
    #     json.dump(results, f, indent=4)  # This ensures new lines for each key-value pair

    # print("Predictions saved to 'my_predictions.json'")

    return results

def get_classifications(request):
    """
    API function for real-time classification of papers from server requests.
    Loads saved model components and processes input data to return predictions.
    """
    # Load model and preprocessing
    model = keras.models.load_model('model.keras')
    with open('vocab.pickle', 'rb') as f:
        vocab = pickle.load(f)
    with open('label_encoder.pickle', 'rb') as f:
        label_encoder = pickle.load(f)
    
    predictions = []
    
    # Handle case where request is a list of papers
    if isinstance(request, list):
        for paper in request:
            if not isinstance(paper, dict):
                continue
                
            # Process formulas
            paper_tokens = []
            for formula in paper.get('formulas', []):
                tokens = tokenize_mathml(formula)
                paper_tokens.extend(tokens)
            
            # Convert to sequence
            sequence = [vocab.get(token, vocab['<UNK>']) for token in paper_tokens]
            
            # Truncate or pad
            if len(sequence) > MAX_SEQ_LENGTH:
                sequence = sequence[:MAX_SEQ_LENGTH]
            else:
                sequence = sequence + [vocab['<PAD>']] * (MAX_SEQ_LENGTH - len(sequence))
            
            # Predict
            pred = model.predict(np.array([sequence]), verbose=0)[0]
            category_idx = np.argmax(pred)
            predictions.append(label_encoder.classes_[category_idx])

    return predictions
#-------------------------------------------------------------------------------------------
"""4. Main Execution"""
#----------------------

def main():
    """
    Command-line interface for training and testing the classifier.
    Parses arguments, manages workflow, and handles model persistence.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='arXiv Math Publication Classifier')
    parser.add_argument('--train', type=str, help='Path to training data file')
    parser.add_argument('--test', type=str, help='Path to test data file')
    parser.add_argument('--output', type=str, default='predictions.json',
                        help='Path to output predictions file (default: predictions.json)')
    
    args = parser.parse_args()
    
    if args.train:
        print("Starting training process...")
        model, vocab, label_encoder = train_and_save_model(args.train)
    else:
        print("Loading pre-trained model...")
        model = keras.models.load_model('model.keras')
        with open('vocab.pickle', 'rb') as f:
            vocab = pickle.load(f)
        with open('label_encoder.pickle', 'rb') as f:
            label_encoder = pickle.load(f)
    
    if args.test:
        print(f"\nProcessing test data from {args.test}")
        results = process_test_data(args.test, vocab, label_encoder, model)
        
        # Save results to predictions.json
        output_file = args.output
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)  # Pretty-printing for better readability
        print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()