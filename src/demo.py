"""
Demo Application for Phase 3
Interactive demonstration of the best trained model.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Suppress TensorFlow warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns


def find_project_root():
    """
    Find the project root directory automatically.
    Works whether you run from project root or src/ directory.
    """
    current = Path(__file__).resolve().parent
    
    # If we're in src/ directory, go up one level
    if current.name == 'src':
        project_root = current.parent
    else:
        project_root = current
    
    # Verify we found the right directory by checking for experiments/
    if not (project_root / 'experiments').exists():
        # Try going up one more level
        project_root = project_root.parent
        if not (project_root / 'experiments').exists():
            print("Warning: Could not auto-detect project root.")
            print(f"Current directory: {current}")
            print(f"Using: {project_root}")
    
    return project_root


# Get project root once at module level
PROJECT_ROOT = find_project_root()


def load_best_model_info():
    """Load information about the best model from experiments."""
    best_info_path = PROJECT_ROOT / 'experiments' / 'best_run_info.json'
    
    if not best_info_path.exists():
        print(f"Error: {best_info_path} not found!")
        print(f"Searched in: {best_info_path.absolute()}")
        print("Please run hyperparameter_search.py first.")
        sys.exit(1)
    
    with open(best_info_path, 'r', encoding='utf-8') as f:
        best_info = json.load(f)
    
    print(f"\n{'='*80}")
    print("BEST MODEL INFORMATION")
    print(f"{'='*80}")
    print(f"Experiment ID: {best_info['experiment_id']}")
    print(f"Timestamp: {best_info['timestamp']}")
    print(f"\nHyperparameters:")
    print(f"  - Hidden Layers: {best_info['hidden_layers']}")
    print(f"  - Learning Rate: {best_info['learning_rate']}")
    print(f"  - Dropout Rate: {best_info['dropout_rate']}")
    print(f"  - Batch Size: {best_info['batch_size']}")
    print(f"  - Activation: {best_info['activation']}")
    print(f"  - Epochs: {best_info['epochs']}")
    print(f"\nPerformance:")
    print(f"  - AUC ROC: {best_info['auc_roc']:.4f}")
    print(f"  - Accuracy: {best_info['accuracy']:.4f}")
    print(f"  - F1 Score: {best_info['f1']:.4f}")
    print(f"  - Precision: {best_info.get('precision', 'N/A')}")
    print(f"  - Recall: {best_info.get('recall', 'N/A')}")
    print(f"{'='*80}\n")
    
    return best_info


def find_model_file(best_info):
    """Find the model file corresponding to the best experiment."""
    # Search in src/output directories (FIXED PATH!)
    output_dir = PROJECT_ROOT / 'src' / 'output'
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        print(f"Project root detected as: {PROJECT_ROOT}")
        sys.exit(1)
    
    # Look for directories matching the timestamp
    timestamp = best_info['timestamp']
    
    print(f"Searching for model file with timestamp: {timestamp}")
    print(f"Looking in: {output_dir}")
    
    # Search all subdirectories
    model_files = list(output_dir.rglob(f'{timestamp}/model_best.keras'))
    
    if not model_files:
        print(f"⚠️  Exact timestamp not found, searching for all models...")
        # Try alternative search - find most recent model
        all_models = list(output_dir.rglob('model_best.keras'))
        if all_models:
            # Sort by modification time, get most recent
            model_files = [max(all_models, key=lambda p: p.stat().st_mtime)]
            print(f"⚠️  Using most recent model")
    
    if not model_files:
        print(f"Error: Could not find model file!")
        print(f"Searched in: {output_dir}")
        print(f"Expected pattern: .../{{timestamp}}/model_best.keras")
        print(f"\nAvailable subdirectories in output/:")
        try:
            subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
            for subdir in subdirs[:10]:  # Show first 10
                print(f"  - {subdir.name}")
        except:
            print("  (Could not list directories)")
        sys.exit(1)
    
    model_path = model_files[0]
    print(f"✅ Found model: {model_path}\n")
    
    return model_path


def load_data():
    """Load and prepare the dataset."""
    data_path = PROJECT_ROOT / 'src' / 'data_sources' / 'liquid_biopsy_data.csv'
    
    if not data_path.exists():
        print(f"Error: {data_path} not found!")
        print(f"Searched in: {data_path.absolute()}")
        print(f"Project root: {PROJECT_ROOT}")
        sys.exit(1)
    
    df = pd.read_csv(data_path, low_memory=False)
    
    # Extract features and target
    X_df = df.iloc[:, 1:-16]
    y_df = df.cancer
    
    # Get feature names
    feature_names = X_df.columns.to_list()
    
    X = X_df.to_numpy()
    y = y_df.to_numpy()
    
    # Split according to dataset columns
    X_train = X[df["isTraining"] == 1]
    y_train = y[df["isTraining"] == 1]
    X_val = X[df["isValidation"] == 1]
    y_val = y[df["isValidation"] == 1]
    X_test = X[df["isTest"] == 1]
    y_test = y[df["isTest"] == 1]
    
    # Scale data (DNN requires scaling)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Data loaded:")
    print(f"   Train: {X_train_scaled.shape}")
    print(f"   Val: {X_val_scaled.shape}")
    print(f"   Test: {X_test_scaled.shape}")
    print(f"   Features: {len(feature_names)}\n")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_names


def predict_sample(model, X, y, sample_idx, feature_names):
    """Predict and explain a single sample."""
    print(f"\n{'='*80}")
    print(f"SAMPLE PREDICTION - Index {sample_idx}")
    print(f"{'='*80}")
    
    # Get sample
    sample = X[sample_idx:sample_idx+1]
    true_label = y[sample_idx]
    
    # Predict
    prediction_prob = model.predict(sample, verbose=0)[0][0]
    prediction_class = 1 if prediction_prob >= 0.5 else 0
    
    # Display results
    print(f"\nTrue Label: {'Cancer' if true_label == 1 else 'No Cancer'} ({true_label})")
    print(f"Predicted: {'Cancer' if prediction_class == 1 else 'No Cancer'} ({prediction_class})")
    print(f"Probability: {prediction_prob:.4f}")
    print(f"Confidence: {max(prediction_prob, 1-prediction_prob):.4f}")
    
    if prediction_class == true_label:
        print("✅ Correct prediction!")
    else:
        print("❌ Incorrect prediction!")
    
    # Show top features with highest values
    print(f"\nTop 10 Feature Values for this sample:")
    sample_features = sample[0]
    top_indices = np.argsort(sample_features)[-10:][::-1]
    
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. {feature_names[idx]}: {sample_features[idx]:.4f}")
    
    print(f"{'='*80}\n")


def evaluate_test_set(model, X_test, y_test):
    """Evaluate model on full test set."""
    print(f"\n{'='*80}")
    print("TEST SET EVALUATION")
    print(f"{'='*80}")
    
    # Predict
    predictions_prob = model.predict(X_test, verbose=0)
    predictions = (predictions_prob >= 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    print(f"\nMetrics:")
    print(f"  - Accuracy:  {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1 Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    TN, FP, FN, TP = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {TN}")
    print(f"  False Positives: {FP}")
    print(f"  False Negatives: {FN}")
    print(f"  True Positives:  {TP}")
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Cancer', 'Cancer'],
                yticklabels=['No Cancer', 'Cancer'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - Test Set', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print(f"\n{'='*80}\n")


def interactive_demo(model, X_test, y_test, feature_names):
    """Interactive demo mode."""
    print(f"\n{'='*80}")
    print("INTERACTIVE DEMO MODE")
    print(f"{'='*80}")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Enter a sample index (0-{len(X_test)-1}) or 'q' to quit")
    print(f"{'='*80}\n")
    
    while True:
        try:
            user_input = input("Sample index: ").strip()
            
            if user_input.lower() == 'q':
                print("Exiting interactive demo.")
                break
            
            sample_idx = int(user_input)
            
            if sample_idx < 0 or sample_idx >= len(X_test):
                print(f"Invalid index. Must be between 0 and {len(X_test)-1}")
                continue
            
            predict_sample(model, X_test, y_test, sample_idx, feature_names)
            
        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")
        except KeyboardInterrupt:
            print("\n\nExiting interactive demo.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Demo application for the best trained cancer detection model"
    )
    parser.add_argument('--sample', type=int, default=None,
                       help='Predict specific test sample by index')
    parser.add_argument('--test_set', action='store_true',
                       help='Evaluate on full test set')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode - predict multiple samples')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"CANCER DETECTION DEMO - Phase 3")
    print(f"{'='*80}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"{'='*80}\n")
    
    # Load best model info
    best_info = load_best_model_info()
    
    # Find and load model
    model_path = find_model_file(best_info)
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!\n")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_data()
    
    # Execute requested operation
    if args.test_set:
        evaluate_test_set(model, X_test, y_test)
    elif args.sample is not None:
        if args.sample < 0 or args.sample >= len(X_test):
            print(f"Error: Sample index must be between 0 and {len(X_test)-1}")
            sys.exit(1)
        predict_sample(model, X_test, y_test, args.sample, feature_names)
    elif args.interactive:
        interactive_demo(model, X_test, y_test, feature_names)
    else:
        # Default: show first 5 samples
        print("No specific operation requested. Showing predictions for first 5 test samples:")
        for i in range(min(5, len(X_test))):
            predict_sample(model, X_test, y_test, i, feature_names)
        
        print("\n" + "="*80)
        print("Usage examples:")
        print("  python src/demo.py --sample 10        # Predict sample #10")
        print("  python src/demo.py --test_set         # Evaluate full test set")
        print("  python src/demo.py --interactive      # Interactive mode")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()
