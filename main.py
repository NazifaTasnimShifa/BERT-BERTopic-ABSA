import pandas as pd
import torch
import numpy as np
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification
)
from train import train_model
from visualize import plot_comparison_charts
from pathlib import Path

# Setup
torch.manual_seed(42)
np.random.seed(42)
output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load processed data
df = pd.read_csv(output_dir / 'processed_data.csv')

# Models
models = {
    'BERT': {'tokenizer': BertTokenizer, 'model': BertForSequenceClassification, 'name': 'bert-base-uncased'},
    'RoBERTa': {'tokenizer': RobertaTokenizer, 'model': RobertaForSequenceClassification, 'name': 'roberta-base'},
    'DistilBERT': {'tokenizer': DistilBertTokenizer, 'model': DistilBertForSequenceClassification, 'name': 'distilbert-base-uncased'},
}

# Train or load models
results = {}
force_retrain = False  # Set to True to force retraining
for name, config in models.items():
    try:
        fold_results, avg_metrics, train_losses, val_losses = train_model(
            config['model'], config['tokenizer'], config['name'], df, device, force_retrain=force_retrain
        )
        results[name] = {
            'true_labels': fold_results[-1]['true_labels'],
            'pred_labels': fold_results[-1]['pred_labels'],
            'avg_metrics': avg_metrics,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    except Exception as e:
        print(f'Error processing {name}: {str(e)}')

# Plot comparisons
plot_comparison_charts(results)

print("Processing complete. Visualizations saved in outputs/")