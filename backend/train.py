"""
Training script for the GxE Deep Learning Model.
Handles training loop, evaluation, and model saving.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import joblib

from model import GxEModel

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__))


def compute_cindex(y_true, y_pred):
    """Compute concordance index."""
    try:
        from lifelines.utils import concordance_index
        return concordance_index(y_true, y_pred)
    except Exception:
        return 0.5


def create_dataloader(X_gene, X_env, X_methyl, y, batch_size=32, shuffle=True):
    """Create PyTorch DataLoader from numpy arrays."""
    dataset = TensorDataset(
        torch.FloatTensor(X_gene),
        torch.FloatTensor(X_env),
        torch.FloatTensor(X_methyl),
        torch.FloatTensor(y),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model():
    """Full training pipeline."""
    print("Loading processed data...")
    data = joblib.load(os.path.join(PROCESSED_DIR, 'processed_data.pkl'))

    # Create dataloaders
    train_loader = create_dataloader(
        data['X_gene_train'], data['X_env_train'],
        data['X_methyl_train'], data['y_train'], batch_size=32
    )
    val_loader = create_dataloader(
        data['X_gene_val'], data['X_env_val'],
        data['X_methyl_val'], data['y_val'], batch_size=32, shuffle=False
    )
    test_loader = create_dataloader(
        data['X_gene_test'], data['X_env_test'],
        data['X_methyl_test'], data['y_test'], batch_size=32, shuffle=False
    )

    # Model dimensions
    n_genes = data['X_gene_train'].shape[1]
    n_env = data['X_env_train'].shape[1]
    n_methyl = data['X_methyl_train'].shape[1]

    print(f"Model dimensions: genes={n_genes}, env={n_env}, methyl={n_methyl}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = GxEModel(n_genes, n_env, n_methyl).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    max_epochs = 100
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\nStarting training for max {max_epochs} epochs...")
    print("-" * 60)

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_losses = []
        train_preds, train_labels = [], []

        for X_g, X_e, X_m, y_batch in train_loader:
            X_g, X_e, X_m, y_batch = X_g.to(device), X_e.to(device), X_m.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_g, X_e, X_m)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())

        # Validate
        model.eval()
        val_losses = []
        val_preds, val_labels = [], []

        with torch.no_grad():
            for X_g, X_e, X_m, y_batch in val_loader:
                X_g, X_e, X_m, y_batch = X_g.to(device), X_e.to(device), X_m.to(device), y_batch.to(device)
                outputs = model(X_g, X_e, X_m)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        # Metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = accuracy_score(train_labels, [1 if p > 0.5 else 0 for p in train_preds])
        val_acc = accuracy_score(val_labels, [1 if p > 0.5 else 0 for p in val_preds])

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['train_acc'].append(float(train_acc))
        history['val_acc'].append(float(val_acc))

        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model and evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating best model on test set...")
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pth'), weights_only=True))
    model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for X_g, X_e, X_m, y_batch in test_loader:
            X_g, X_e, X_m, y_batch = X_g.to(device), X_e.to(device), X_m.to(device), y_batch.to(device)
            outputs = model(X_g, X_e, X_m)
            test_preds.extend(outputs.cpu().numpy())
            test_labels.extend(y_batch.cpu().numpy())

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_binary = (test_preds > 0.5).astype(int)

    accuracy = float(accuracy_score(test_labels, test_binary))
    try:
        auc_roc = float(roc_auc_score(test_labels, test_preds))
    except ValueError:
        auc_roc = 0.5
    f1 = float(f1_score(test_labels, test_binary, zero_division=0))
    c_index = float(compute_cindex(test_labels, test_preds))

    metrics = {
        'accuracy': round(accuracy, 4),
        'auc_roc': round(auc_roc, 4),
        'f1_score': round(f1, 4),
        'c_index': round(c_index, 4),
    }

    print(f"\nTest Results:")
    print(f"   Accuracy:  {metrics['accuracy']}")
    print(f"   AUC-ROC:   {metrics['auc_roc']}")
    print(f"   F1-Score:  {metrics['f1_score']}")
    print(f"   C-index:   {metrics['c_index']}")

    # Save metrics and history
    with open(os.path.join(PROCESSED_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(PROCESSED_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Save model config
    model_config = {
        'n_genes': n_genes,
        'n_env': n_env,
        'n_methyl': n_methyl,
    }
    with open(os.path.join(PROCESSED_DIR, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)

    print(f"\n✅ Training complete! Model saved to {MODEL_DIR}/best_model.pth")
    return model, metrics, history


if __name__ == '__main__':
    train_model()
