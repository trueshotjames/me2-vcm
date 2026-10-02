# training_env_wake/train.py

import torch
import torch.nn as nn
import argparse
import os
import time
from tqdm import tqdm
from model import WakeWordModel
from dataset_loader import WakeWordDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='e.g., cuda, cuda:0, or 0,1. Defaults to CPU if no GPU is found.')
    parser.add_argument('--batches', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    return parser.parse_args()

def compute_binary_metrics(scores, labels):
    """Binary accuracy and false-accept rate for wake-word.

    Args:
        scores: tensor of shape (N,) - sigmoid probabilities
        labels: tensor of shape (N,) - binary labels (0 or 1)

    Returns:
        (accuracy, far, tp, fp, tn, fn)
        - accuracy = (TP + TN) / total in [0, 1]
        - far = FP / max(1, FP + TN) in [0, 1]
    """
    preds = (scores >= 0.5).long()
    labels = labels.long()
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return accuracy, far, tp, fp, tn, fn


def train():
    args = get_args()
    
    # Robust CPU Fallback
    if not torch.cuda.is_available() or args.device.lower() == 'cpu':
        print("No GPU detected or CPU explicitly requested. Defaulting to CPU.")
        device = torch.device('cpu')
    else:
        if ',' in args.device or args.device.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
            device = torch.device('cuda')
        else:
            device = torch.device(args.device)

    # Initialize running tracker (append if exists so history is not lost)
    file_exists = os.path.exists("wake_training_results.txt")
    results_f = open("wake_training_results.txt", "a")
    if not file_exists:
        results_f.write("epoch\ttrain_loss\tval_f1\tval_accuracy\tval_far\truntime\n")

    model = WakeWordModel().to(device)

    # Automatically retrieve the saved model if it exists
    if os.path.exists("wake_word_model.pth"):
        model.load_state_dict(torch.load("wake_word_model.pth", map_location=device))
        print("Successfully loaded existing wake_word_model.pth. Resuming training...")
    
    # BCELoss replaces BCEWithLogitsLoss because Sigmoid is now built into the model
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_data = WakeWordDataset("dataset/wake_train_data.csv")
    val_data = WakeWordDataset("dataset/wake_val_data.csv")
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batches, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batches, shuffle=False)
    
    best_val_f1 = -1.0
    epochs_no_improve = 0
    best_epoch = 0

    print(f"Starting Binary Wake Word Training on {device}...")
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # --- TRAINING PHASE ---
        model.train()
        total_loss = 0.0
        
        # Wrap the loader in tqdm and set leave=False so it clears after each epoch
        batch_loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}] Train", leave=False)
        
        for audio, labels in batch_loop:
            audio, labels = audio.to(device), labels.to(device)
            optimizer.zero_grad()
            
            output = model(audio).squeeze(-1)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Update the progress bar to show the current batch loss on the right
            batch_loop.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_train_loss = total_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval()
        tp = fp = fn = 0
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for audio, labels in val_loader:
                audio, labels = audio.to(device), labels.to(device)
                output = model(audio).squeeze(-1)
                # Collect raw sigmoid scores/labels for accuracy + FAR
                all_scores.append(output.detach().cpu())
                all_labels.append(labels.detach().cpu())
                
                # Threshold the baked-in Sigmoid output at 0.5
                preds = (output > 0.5).float()
                
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
                
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        val_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        all_scores_t = torch.cat(all_scores)
        all_labels_t = torch.cat(all_labels)
        val_accuracy, val_far, _tp, _fp, _tn, _fn = compute_binary_metrics(all_scores_t, all_labels_t)
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch:03d} Summary: Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_accuracy:.4f} | Val FAR: {val_far:.4f} | Runtime: {epoch_duration:.2f}s")

        results_f.write(f"{epoch}\t{avg_train_loss:.6f}\t{val_f1:.6f}\t{val_accuracy:.6f}\t{val_far:.6f}\t{epoch_duration:.2f}\n")
        results_f.flush()
        
        # --- CHECKPOINT & EARLY STOPPING ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), "wake_word_model.pth")
            print(f"--- New Best Model Saved at Epoch {epoch} with F1: {best_val_f1:.4f} ---")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break
            
    results_f.close()
    print(f"Training Finished. Best Epoch: {best_epoch}, Best Val F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    train()