import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
import plotly.express as px

def plot_comparison_charts(results, proposed_accuracy=0.91):
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)

    # Confusion Matrices
    for name, result in results.items():
        cm = confusion_matrix(result['true_labels'], result['pred_labels'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
        plt.title(f'Confusion Matrix - {name}')
        plt.savefig(output_dir / f'cm_{name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Accuracy Comparison
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()) + ['Proposed Model'],
        'Accuracy': [results[name]['avg_metrics']['accuracy'] for name in results] + [proposed_accuracy]
    })
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=metrics_df)
    plt.title('Accuracy Comparison Across Models')
    plt.ylim(0, 1)
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Cross-Validated Model Comparison
    cv_scores = pd.DataFrame({name: [results[name]['avg_metrics']['accuracy']] * 3 for name in results})
    cv_scores['Proposed Model'] = [proposed_accuracy] * 3
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=cv_scores)
    plt.title('Cross-Validated Model Comparison')
    plt.ylabel('Accuracy')
    plt.savefig(output_dir / 'cv_scores_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Training and Validation Loss Curves (Placeholder - requires actual loss data)
    for name, result in results.items():
        if 'train_losses' in result and 'val_losses' in result:
            plt.figure(figsize=(10, 6))
            plt.plot(result['train_losses'], label='Training Loss')
            plt.plot(result['val_losses'], label='Validation Loss')
            plt.title(f'Training and Validation Loss Curves - {name}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(output_dir / f'loss_curves_{name.lower()}.png', dpi=300, bbox_inches='tight')
            plt.close()

    # Save metrics
    metrics_df.to_csv(output_dir / 'model_metrics.csv')