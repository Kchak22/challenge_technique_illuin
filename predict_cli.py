# predict_cli.py
import argparse
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Configuration
TARGET_TAGS = ['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']
CODEBERT_MODEL = 'microsoft/codebert-base'
MAX_CODE_LENGTH = 512

def preprocess_with_tokens(text):
    """Preprocess text with tokenization"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\$\$\$.*?\$\$\$', ' mathformulatoken ', text)
    text = re.sub(r'\b[a-z]_\w+\b', ' indexedvariabletoken ', text)
    text = re.sub(r'\b[a-zA-Z]\b', ' variabletoken ', text)
    text = re.sub(r'\b\d+\b', ' numbertoken ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    filtered_words = [word for word in words if len(word) > 2]
    return " ".join(filtered_words)

def load_json_files(path):
    """Load JSON file(s) from path (file or directory)"""
    path = Path(path)
    all_data = []
    
    if path.is_file():
        # Single file
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                all_data = [data]
            else:
                all_data = data
    elif path.is_dir():
        # Directory of JSON files
        json_files = sorted(path.glob("*.json"))
        print(f"Found {len(json_files)} JSON files in directory")
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data['_source_file'] = json_file.name
                        all_data.append(data)
                    elif isinstance(data, list):
                        for item in data:
                            item['_source_file'] = json_file.name
                        all_data.extend(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    else:
        raise ValueError(f"Path {path} is neither a file nor a directory")
    
    return all_data

def load_and_preprocess_data(path):
    """Load and preprocess data from JSON file(s)"""
    data = load_json_files(path)
    df = pd.DataFrame(data)
    
    # Preprocess prob_desc_notes
    if 'prob_desc_notes' in df.columns:
        df['prob_desc_notes'] = df['prob_desc_notes'].fillna('')
    else:
        df['prob_desc_notes'] = ''
    
    # Handle sample inputs/outputs
    if 'prob_desc_sample_inputs' in df.columns:
        df['prob_desc_sample_inputs'] = df['prob_desc_sample_inputs'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else (x if isinstance(x, str) else '')
        )
    
    if 'prob_desc_sample_outputs' in df.columns:
        df['prob_desc_sample_outputs'] = df['prob_desc_sample_outputs'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else (x if isinstance(x, str) else '')
        )
    
    # Combine description with notes
    if 'prob_desc_description' not in df.columns:
        df['prob_desc_description'] = ''
    
    df['prob_desc_description'] = df['prob_desc_description'].fillna('') + ' ' + df['prob_desc_notes']
    
    # Preprocess text
    df['prob_desc_description'] = df['prob_desc_description'].apply(preprocess_with_tokens)
    
    # Handle difficulty
    if 'difficulty' in df.columns:
        df['difficulty_clean'] = df['difficulty'].replace(-1, np.nan)
    else:
        df['difficulty_clean'] = np.nan
    
    # Handle source code
    if 'source_code' not in df.columns:
        df['source_code'] = ''
    df['source_code'] = df['source_code'].fillna('')
    
    # Handle tags for evaluation
    if 'tags' in df.columns:
        df['tags_list'] = df['tags'].apply(
            lambda x: sorted([tag.strip() for tag in (x.split(",") if isinstance(x, str) else x) if tag.strip() in TARGET_TAGS])
        )
        for tag in TARGET_TAGS:
            df[f'label_{tag}'] = df['tags_list'].apply(lambda x: 1 if tag in x else 0)
    
    return df

def extract_codebert_embeddings(codes, device=None, batch_size=32):
    """Extract CodeBERT embeddings for source code"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"  Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
    codebert = AutoModel.from_pretrained(CODEBERT_MODEL).to(device)
    codebert.eval()
    
    embeddings = []
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i+batch_size]
        batch_emb = []
        
        for code in batch:
            if len(code.strip()) > 0:
                inputs = tokenizer(code, max_length=MAX_CODE_LENGTH, truncation=True,
                                 padding='max_length', return_tensors='pt').to(device)
                with torch.no_grad():
                    outputs = codebert(**inputs)
                    emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
            else:
                emb = np.zeros(768)
            batch_emb.append(emb)
        
        embeddings.extend(batch_emb)
        
        if (i + batch_size) % 100 == 0:
            print(f"    Processed {min(i + batch_size, len(codes))}/{len(codes)} samples")
    
    return np.array(embeddings)

def train_model(train_data_path, output_model_path, model_type='text'):
    """Train a model (text-only or multimodal)"""
    print(f"Training {model_type} model...")
    
    # Load training data
    df = pd.read_parquet(train_data_path)
    
    # Prepare labels
    df["tags_list"] = df["tags_list"].apply(list)
    for tag in TARGET_TAGS:
        df[f'label_{tag}'] = df['tags_list'].apply(
            lambda x: 1 if isinstance(x, list) and tag in x else 0
        )
    
    y = df[[f'label_{tag}' for tag in TARGET_TAGS]].values
    
    # Preprocess text if not already done
    if not df['prob_desc_description'].iloc[0] or 'token' not in str(df['prob_desc_description'].iloc[0]):
        print("Preprocessing text...")
        df['prob_desc_description'] = df['prob_desc_description'].apply(preprocess_with_tokens)
    
    # TF-IDF
    print("Extracting TF-IDF features...")
    stop_words = list(ENGLISH_STOP_WORDS) + ['problem', 'input', 'output', 'test', 'case', 'example', 'note', 'notein']
    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.7,
        stop_words=stop_words
    )
    X_text = tfidf.fit_transform(df['prob_desc_description'].fillna(''))
    
    # Difficulty
    df['difficulty_clean'] = df['difficulty'].replace(-1, np.nan)
    median_diff = df['difficulty_clean'].median()
    df['difficulty_clean'] = df['difficulty_clean'].fillna(median_diff)
    
    scaler = StandardScaler()
    X_diff = scaler.fit_transform(df['difficulty_clean'].values.reshape(-1, 1))
    
    # Combine features
    from scipy.sparse import hstack, csr_matrix
    
    if model_type == 'multimodal':
        # Extract CodeBERT embeddings
        print("Extracting CodeBERT embeddings...")
        X_code = extract_codebert_embeddings(df['source_code'].values)
        X = hstack([X_text, csr_matrix(X_code), csr_matrix(X_diff)])
        print(f"Total features: {X.shape[1]} (TF-IDF: {X_text.shape[1]}, CodeBERT: {X_code.shape[1]}, Difficulty: 1)")
    else:
        X = hstack([X_text, csr_matrix(X_diff)])
        print(f"Total features: {X.shape[1]} (TF-IDF: {X_text.shape[1]}, Difficulty: 1)")
    
    # Train model
    print("Training XGBoost model...")
    base_clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        tree_method='hist'
    )
    
    model = MultiOutputClassifier(base_clf)
    model.fit(X, y)
    
    # Save model and preprocessors
    model_data = {
        'model': model,
        'tfidf': tfidf,
        'scaler': scaler,
        'median_difficulty': median_diff,
        'target_tags': TARGET_TAGS,
        'model_type': model_type
    }
    
    with open(output_model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {output_model_path}")

def predict(input_path, model_path):
    """Make predictions on a single sample or dataset"""
    start_time = time.time()
    
    # Load model
    print("Loading model...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    tfidf = model_data['tfidf']
    scaler = model_data['scaler']
    median_difficulty = model_data['median_difficulty']
    model_type = model_data.get('model_type', 'text')
    
    print(f"Model type: {model_type}")
    
    # Load and preprocess data
    print("Loading data...")
    df = load_and_preprocess_data(input_path)
    
    # Extract features
    print("Extracting features...")
    X_text = tfidf.transform(df['prob_desc_description'].fillna(''))
    
    # Difficulty
    df['difficulty_clean'] = df['difficulty_clean'].fillna(median_difficulty)
    X_diff = scaler.transform(df['difficulty_clean'].values.reshape(-1, 1))
    
    # Combine
    from scipy.sparse import hstack, csr_matrix
    
    if model_type == 'multimodal':
        print("Extracting CodeBERT embeddings...")
        X_code = extract_codebert_embeddings(df['source_code'].values)
        X = hstack([X_text, csr_matrix(X_code), csr_matrix(X_diff)])
    else:
        X = hstack([X_text, csr_matrix(X_diff)])
    
    # Predict
    print("Making predictions...")
    y_pred = model.predict(X)
    y_proba = np.column_stack([
        estimator.predict_proba(X)[:, 1] for estimator in model.estimators_
    ])
    
    elapsed_time = time.time() - start_time
    
    # Format results
    results = []
    for i in range(len(df)):
        sample_result = {
            'sample_index': i,
            'predicted_tags': [TARGET_TAGS[j] for j in range(len(TARGET_TAGS)) if y_pred[i, j] == 1],
            'probabilities': {TARGET_TAGS[j]: float(y_proba[i, j]) for j in range(len(TARGET_TAGS))}
        }
        if '_source_file' in df.columns:
            sample_result['source_file'] = df['_source_file'].iloc[i]
        results.append(sample_result)
    
    print(f"\nPrediction completed in {elapsed_time:.2f} seconds ({elapsed_time/len(df):.2f}s per sample)")
    
    return results, df, y_pred, y_proba

def precision_at_k(y_true, y_proba, k=3):
    """Calculate Precision@K"""
    precisions = []
    for i in range(len(y_true)):
        top_k_indices = np.argsort(y_proba[i])[::-1][:k]
        top_k_pred = np.zeros(len(y_true[i]))
        top_k_pred[top_k_indices] = 1
        
        true_positives = np.sum(top_k_pred * y_true[i])
        precision = true_positives / k if k > 0 else 0
        precisions.append(precision)
    
    return np.mean(precisions)

def recall_at_k(y_true, y_proba, k=3):
    """Calculate Recall@K"""
    recalls = []
    for i in range(len(y_true)):
        top_k_indices = np.argsort(y_proba[i])[::-1][:k]
        top_k_pred = np.zeros(len(y_true[i]))
        top_k_pred[top_k_indices] = 1
        
        true_positives = np.sum(top_k_pred * y_true[i])
        actual_positives = np.sum(y_true[i])
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        recalls.append(recall)
    
    return np.mean(recalls)

def jaccard_score_per_sample(y_true, y_pred):
    """Calculate per-sample Jaccard similarity and return mean"""
    jaccard_scores = []
    for i in range(len(y_true)):
        intersection = np.sum(y_true[i] * y_pred[i])
        union = np.sum((y_true[i] + y_pred[i]) > 0)
        jaccard = intersection / union if union > 0 else 0
        jaccard_scores.append(jaccard)
    
    return np.mean(jaccard_scores), jaccard_scores

def evaluate(input_path, model_path):
    """Evaluate model on a test dataset"""
    from sklearn.metrics import (
        classification_report, f1_score, average_precision_score,
        hamming_loss, roc_auc_score, precision_score, recall_score
    )
    
    # Get predictions
    results, df, y_pred, y_proba = predict(input_path, model_path)
    
    # Check if ground truth is available
    if 'tags' not in df.columns:
        print("Error: Ground truth tags not found in the data. Cannot evaluate.")
        return
    
    # Extract ground truth
    y_true = df[[f'label_{tag}' for tag in TARGET_TAGS]].values
    
    # Calculate metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Overall metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Mean Average Precision
    mean_ap = np.mean([
        average_precision_score(y_true[:, i], y_proba[:, i])
        for i in range(len(TARGET_TAGS)) if y_true[:, i].sum() > 0
    ])
    
    # Macro AUC (ROC)
    try:
        macro_auc = np.mean([
            roc_auc_score(y_true[:, i], y_proba[:, i])
            for i in range(len(TARGET_TAGS)) if len(np.unique(y_true[:, i])) > 1
        ])
    except:
        macro_auc = np.nan
    
    # Hamming Loss
    hamming = hamming_loss(y_true, y_pred)
    
    # Jaccard Similarity
    mean_jaccard, _ = jaccard_score_per_sample(y_true, y_pred)
    
    # Precision@K and Recall@K
    precision_at_1 = precision_at_k(y_true, y_proba, k=1)
    recall_at_1 = recall_at_k(y_true, y_proba, k=1)
    precision_at_3 = precision_at_k(y_true, y_proba, k=3)
    recall_at_3 = recall_at_k(y_true, y_proba, k=3)
    precision_at_5 = precision_at_k(y_true, y_proba, k=5)
    recall_at_5 = recall_at_k(y_true, y_proba, k=5)
    
    print(f"\n{'='*80}")
    print(f"OVERALL METRICS")
    print(f"{'='*80}")
    print(f"\nF1-Scores:")
    print(f"  Macro F1:            {macro_f1:.4f}")
    print(f"  Micro F1:            {micro_f1:.4f}")
    print(f"  Weighted F1:         {weighted_f1:.4f}")
    
    print(f"\nPrecision & Recall:")
    print(f"  Macro Precision:     {macro_precision:.4f}")
    print(f"  Macro Recall:        {macro_recall:.4f}")
    
    print(f"\nProbability-Based Metrics:")
    print(f"  Mean Avg Precision:  {mean_ap:.4f}")
    print(f"  Macro AUC (ROC):     {macro_auc:.4f}")
    
    print(f"\nSet-Based Metrics:")
    print(f"  Hamming Loss:        {hamming:.4f}")
    print(f"  Jaccard Similarity:  {mean_jaccard:.4f}")
    
    print(f"\nTop-K Metrics:")
    print(f"  Precision@1:         {precision_at_1:.4f}")
    print(f"  Recall@1:            {recall_at_1:.4f}")
    print(f"  Precision@3:         {precision_at_3:.4f}")
    print(f"  Recall@3:            {recall_at_3:.4f}")
    print(f"  Precision@5:         {precision_at_5:.4f}")
    print(f"  Recall@5:            {recall_at_5:.4f}")
    
    print(f"\n{'='*80}")
    print(f"PER-TAG PERFORMANCE")
    print(f"{'='*80}")
    print(classification_report(y_true, y_pred, target_names=TARGET_TAGS, digits=3, zero_division=0))
    
    # Per-tag metrics table
    print(f"\n{'='*80}")
    print(f"PER-TAG DETAILED METRICS")
    print(f"{'='*80}")
    print(f"{'Tag':<20} {'Support':<10} {'F1':<8} {'Precision':<10} {'Recall':<10} {'AP':<8}")
    print("-" * 80)
    
    for i, tag in enumerate(TARGET_TAGS):
        support = int(y_true[:, i].sum())
        tag_f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        tag_precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        tag_recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        tag_ap = average_precision_score(y_true[:, i], y_proba[:, i]) if support > 0 else 0
        
        print(f"{tag:<20} {support:<10} {tag_f1:<8.3f} {tag_precision:<10.3f} {tag_recall:<10.3f} {tag_ap:<8.3f}")
    
    # Show some examples
    print(f"\n{'='*80}")
    print(f"EXAMPLE PREDICTIONS")
    print(f"{'='*80}")
    
    # Perfect predictions
    perfect_mask = np.all(y_true == y_pred, axis=1)
    if perfect_mask.sum() > 0:
        print("\nPerfect Predictions (first 2):")
        for idx in np.where(perfect_mask)[0][:2]:
            print(f"\n  Sample {idx}:")
            if '_source_file' in df.columns:
                print(f"    Source: {df['_source_file'].iloc[idx]}")
            print(f"    True tags:      {df['tags_list'].iloc[idx]}")
            print(f"    Predicted tags: {results[idx]['predicted_tags']}")
    
    # Worst predictions
    error_counts = np.sum(y_true != y_pred, axis=1)
    worst_indices = np.argsort(error_counts)[::-1][:2]
    print("\nWorst Predictions (first 2):")
    for idx in worst_indices:
        print(f"\n  Sample {idx} ({error_counts[idx]} errors):")
        if '_source_file' in df.columns:
            print(f"    Source: {df['_source_file'].iloc[idx]}")
        print(f"    True tags:      {df['tags_list'].iloc[idx]}")
        print(f"    Predicted tags: {results[idx]['predicted_tags']}")
        print(f"    Top probabilities:")
        sorted_probs = sorted(results[idx]['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        for tag, prob in sorted_probs:
            print(f"      {tag:<15s}: {prob:.4f}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total samples:                 {len(y_true)}")
    print(f"Perfect predictions:           {perfect_mask.sum()} ({perfect_mask.sum()/len(y_true)*100:.1f}%)")
    print(f"Samples with 0 errors:         {np.sum(error_counts == 0)} ({np.sum(error_counts == 0)/len(y_true)*100:.1f}%)")
    print(f"Samples with 1 error:          {np.sum(error_counts == 1)} ({np.sum(error_counts == 1)/len(y_true)*100:.1f}%)")
    print(f"Samples with 2+ errors:        {np.sum(error_counts >= 2)} ({np.sum(error_counts >= 2)/len(y_true)*100:.1f}%)")
    print(f"Mean errors per sample:        {error_counts.mean():.2f}")
    print(f"Max errors in a sample:        {error_counts.max()}")
    
    avg_tags_true = np.sum(y_true, axis=1).mean()
    avg_tags_pred = np.sum(y_pred, axis=1).mean()
    print(f"\nAverage tags per sample (true): {avg_tags_true:.2f}")
    print(f"Average tags per sample (pred): {avg_tags_pred:.2f}")

def main():
    parser = argparse.ArgumentParser(
        description='Code Classification Prediction CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a text-only model (fast predictions, <10s)
  python predict_cli.py train --train-data data/preprocessed_dataset.parquet --output models/text_model.pkl --type text
  
  # Train a multimodal model (slower but more accurate)
  python predict_cli.py train --train-data data/preprocessed_dataset.parquet --output models/multimodal_model.pkl --type multimodal
  
  # Make predictions on a single sample
  python predict_cli.py predict --input data/code_classification_dataset/sample_0.json --model models/text_model.pkl
  
  # Make predictions on a directory of JSON files
  python predict_cli.py predict --input data/code_classification_dataset/ --model models/multimodal_model.pkl --output predictions.json
  
  # Evaluate on test set (single file or directory)
  python predict_cli.py evaluate --input data/test_samples.json --model models/text_model.pkl
  python predict_cli.py evaluate --input data/test_dir/ --model models/multimodal_model.pkl
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--train-data', required=True, help='Path to training data (parquet)')
    train_parser.add_argument('--output', required=True, help='Output path for trained model')
    train_parser.add_argument('--type', choices=['text', 'multimodal'], default='text',
                            help='Model type: text (fast, <10s per prediction) or multimodal (slower, more accurate)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--input', required=True, help='Input JSON file or directory')
    predict_parser.add_argument('--model', required=True, help='Path to trained model')
    predict_parser.add_argument('--output', help='Output JSON file for predictions (optional)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on test set')
    eval_parser.add_argument('--input', required=True, help='Input JSON file or directory with ground truth')
    eval_parser.add_argument('--model', required=True, help='Path to trained model')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args.train_data, args.output, args.type)
    
    elif args.command == 'predict':
        results, df, y_pred, y_proba = predict(args.input, args.model)
        
        # Print results
        print("\n" + "="*80)
        print("PREDICTIONS")
        print("="*80)
        print(f"\nTotal samples: {len(results)}")
        
        # Show first few predictions
        for i, result in enumerate(results[:5]):
            print(f"\nSample {result['sample_index'] + 1}:")
            if 'source_file' in result:
                print(f"  Source: {result['source_file']}")
            print(f"  Predicted tags: {result['predicted_tags'] if result['predicted_tags'] else '(none)'}")
            print(f"  Top 3 probabilities:")
            sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
            for tag, prob in sorted_probs:
                print(f"    {tag:<15s}: {prob:.4f}")
        
        if len(results) > 5:
            print(f"\n... and {len(results) - 5} more samples")
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nAll predictions saved to {args.output}")
    
    elif args.command == 'evaluate':
        evaluate(args.input, args.model)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()