import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import numpy as np
from pathlib import Path
import os
from dataset import ABSADataset

def train_model(model_class, tokenizer_class, model_name, df, device, epochs=3, lr=2e-5, folds=3, force_retrain=False):
    output_dir = Path('outputs')
    model_base_dir = output_dir / 'models' / model_name
    model_base_dir.mkdir(parents=True, exist_ok=True)
    
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    fold_results = []
    train_losses = []
    val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        model_dir = model_base_dir / f'fold_{fold}'
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        tokenizer = tokenizer_class.from_pretrained(model_name)
        # Check if model exists and load if not forcing retrain
        if model_dir.exists() and not force_retrain:
            print(f"Loading saved model from {model_dir}")
            model = model_class.from_pretrained(model_dir).to(device)
            tokenizer = tokenizer_class.from_pretrained(model_dir)
        else:
            print(f"Training new model for fold {fold}")
            model = model_class.from_pretrained(model_name, num_labels=3).to(device)

            train_dataset = ABSADataset(train_df['cleaned_text'], train_df['aspects'], train_df['label'], tokenizer)
            val_dataset = ABSADataset(val_df['cleaned_text'], val_df['aspects'], val_df['label'], tokenizer)
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=8)

            optimizer = AdamW(model.parameters(), lr=lr)
            total_steps = len(train_loader) * epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            for epoch in range(epochs):
                model.train()
                total_train_loss = 0
                for batch in train_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    total_train_loss += loss.item()
                avg_train_loss = total_train_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = model(**batch)
                        loss = outputs.loss
                        total_val_loss += loss.item()
                    avg_val_loss = total_val_loss / len(val_loader)
                    val_losses.append(avg_val_loss)

            # Save model and tokenizer
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"Model saved to {model_dir}")

        model.eval()
        true_labels, pred_labels = [], []
        with torch.no_grad():
            val_dataset = ABSADataset(val_df['cleaned_text'], val_df['aspects'], val_df['label'], tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=8)
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                _, preds = torch.max(outputs.logits, 1)
                true_labels.extend(batch['labels'].cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())
        report = classification_report(true_labels, pred_labels, target_names=['negative', 'neutral', 'positive'], output_dict=True, zero_division=0)
        fold_results.append({'true_labels': true_labels, 'pred_labels': pred_labels, 'report': report, 'val_df': val_df})

    avg_accuracy = np.mean([r['report']['accuracy'] for r in fold_results])
    avg_f1 = np.mean([r['report']['macro avg']['f1-score'] for r in fold_results])
    return fold_results, {'accuracy': avg_accuracy, 'f1_macro': avg_f1}, train_losses, val_losses