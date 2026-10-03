# train.py

import torch
from model import CNNVoiceCommandModel
from dataset_loader import VoiceCommandDataset
import torch.nn as nn
import argparse
import os
import time
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='e.g., cuda, cuda:0, or 0,1 for Multi-GPU')
    parser.add_argument('--batches', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_epochs', type=int, default=50)
    return parser.parse_args()

# Helper function to bundle varying label lengths together in a batch
def custom_collate(batch):
    audios = torch.stack([item[0] for item in batch])
    weights = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    # Labels have different lengths, so we keep them in a list for now
    labels = [item[1] for item in batch]
    return audios, labels, weights

def decode_ctc(preds, blank=0):
    # Decodes raw continuous timeline into flat label indices
    decoded = []
    prev = -1
    for p in preds:
        p_val = p.item()
        if p_val != prev and p_val != blank:
            decoded.append(p_val)
        prev = p_val
    return decoded

def calculate_macro_f1(true_labels_list, pred_labels_list, num_classes=57):
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    
    for true_arr, pred_arr in zip(true_labels_list, pred_labels_list):
        for c in range(num_classes):
            true_count = true_arr.count(c)
            pred_count = pred_arr.count(c)
            
            tp[c] += min(true_count, pred_count)
            if pred_count > true_count:
                fp[c] += (pred_count - true_count)
            if true_count > pred_count:
                fn[c] += (true_count - pred_count)
                
    f1_scores = []
    for c in range(num_classes):
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        
    return sum(f1_scores) / num_classes

def compute_extra_metrics(logits, labels):
    """Token-level accuracy and per-class false-accept rate for VCM.

    Args:
        logits: (T, B, C) - CTC logits (post log_softmax), already permuted.
        labels: list of length B - variable-length CTC target label lists.

    Returns:
        (accuracy, mean_far, per_class_far_dict)
        - accuracy: float in [0, 1], correct tokens / total tokens
        - mean_far: float in [0, 1], mean of per-class FAR across all vocab classes
        - per_class_far_dict: {class_idx: far_value, ...} for ALL vocab classes
    """
    preds = logits.argmax(dim=-1)  # (T, B)
    T, B, C = logits.shape
    correct = 0
    total = 0
    # Per-class confusion counters: a token predicted at a frame where the target
    # held a *different* token is a false accept for the predicted class.
    fp_count = [0] * C
    tn_count = [0] * C
    for b in range(B):
        target = labels[b]
        if len(target) == 0:
            continue
        for t in range(T):
            p = int(preds[t, b].item())
            for tg in target:
                if p != tg:
                    fp_count[tg] += 1
                    tn_count[p] += 1
        # Token-level accuracy: does ANY frame predict this exact token?
        for tg in target:
            total += 1
            if bool(preds[:, b].eq(tg).any().item()):
                correct += 1
    accuracy = correct / total if total > 0 else 0.0
    per_class = {}
    for c in range(C):
        denom = fp_count[c] + tn_count[c]
        per_class[c] = fp_count[c] / denom if denom > 0 else 0.0
    mean_far = sum(per_class.values()) / C if C > 0 else 0.0
    return accuracy, mean_far, per_class

def train():
    args = get_args()
    
    # Automatically fallback to CPU if a GPU is requested but not installed
    if 'cuda' in args.device and not torch.cuda.is_available():
        print("Warning: CUDA not detected. Falling back to CPU.")
        device = torch.device('cpu')
    elif ',' in args.device or args.device.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = torch.device('cuda')
    else:
        device = torch.device(args.device)
        
    # Count the words to set the model size
    with open("vocab.txt", "r") as f:
        vocab_size = len([line for line in f.read().splitlines() if line.strip()])
        
    # Pass the flexible size to the model
    model = CNNVoiceCommandModel(num_classes=vocab_size).to(device)

    # Automatically retrieve the saved model if it exists
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        print("Successfully loaded existing best_model.pth. Resuming training...")
    
    # Load Training Data
    standard_train = VoiceCommandDataset("dataset/train_data.csv")
    asian_train = VoiceCommandDataset("dataset/asian_train_data.csv")
    train_dataset = torch.utils.data.ConcatDataset([standard_train, asian_train])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batches, shuffle=True, collate_fn=custom_collate)
    
    # NEW: Load Validation Data side-by-side
    standard_val = VoiceCommandDataset("dataset/val_data.csv")
    asian_val = VoiceCommandDataset("dataset/asian_val_data.csv")
    val_dataset = torch.utils.data.ConcatDataset([standard_val, asian_val])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batches, shuffle=False, collate_fn=custom_collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    # Tell the loss function that the blank token is the last item in vocab.txt
    blank_index = vocab_size - 1
    ctc_loss = nn.CTCLoss(blank=blank_index, reduction='none')

    best_val_f1 = -1.0
    epochs_no_improve = 0
    best_epoch = 0
    results_file = "training_results.txt"

    # Initialize running tracker (append if exists so history is not lost)
    file_exists = os.path.exists(results_file)
    with open(results_file, "a") as f:
        if not file_exists:
            f.write("Epoch\tTrain Loss\tVal F1-Score\tVal Accuracy\tVal FAR\tRuntime\n")

    print(f"Starting training on {args.device} for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time() # NEW
        
        # --- TRAINING PHASE ---
        model.train()
        total_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch_idx, (audio, labels, weights) in enumerate(train_pbar, 1):
            audio = audio.unsqueeze(1).to(device)
            weights = weights.to(device)
            
            optimizer.zero_grad()
            output = model(audio)
            
            # The Conv1d model outputs (Batch, Classes, Time). 
            # We calculate probabilities over the Classes dimension (dim=1).
            output = torch.nn.functional.log_softmax(output, dim=1)
            
            # Reformat to (Time, Batch, Classes) which is strictly required by CTCLoss
            output = output.permute(2, 0, 1) 
            
            input_lengths = torch.full(size=(audio.size(0),), fill_value=output.size(0), dtype=torch.long).to(device)
            target_lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long).to(device)
            flat_labels = torch.cat(labels).to(device)
            
            raw_losses = ctc_loss(output, flat_labels, input_lengths, target_lengths)
            weighted_losses = raw_losses * weights
            loss = weighted_losses.mean()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Live-update the progress bar with the current loss
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")   
            
        avg_train_loss = total_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval()
        all_true_labels = []
        all_pred_labels = []
        all_val_logits = []
        
        with torch.no_grad():
            for audio, labels, _ in val_loader:
                audio = audio.unsqueeze(1).to(device)
                output = model(audio)
                
                all_val_logits.append(output.detach().cpu())
                # Take the highest probability guess for each time step across the classes
                preds = torch.argmax(output, dim=1) 
                
                for i in range(preds.size(0)):
                    # Pass the new blank index so the decoder knows what to discard
                    decoded = decode_ctc(preds[i], blank=vocab_size - 1)
                    all_pred_labels.append(decoded)
                    all_true_labels.append(labels[i].tolist())
                    
        val_f1 = calculate_macro_f1(all_true_labels, all_pred_labels, num_classes=vocab_size)

        # Token-level accuracy + per-class false-accept rate (full-epoch tensors).
        # all_val_logits entries are (B, C, T); cat -> (B, C, T) -> permute -> (T, B, C).
        val_logits = torch.cat(all_val_logits, dim=0).permute(2, 0, 1)
        val_accuracy, val_far, per_class_far = compute_extra_metrics(
            val_logits, all_true_labels
        )

        # NEW: Stop timer and print the full epoch summary
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch} Summary: Avg Train Loss: {avg_train_loss:.4f} | Val F1-Score: {val_f1:.4f} | Val Acc: {val_accuracy:.4f} | Val False Alarm Rate: {val_far:.4f} | Runtime: {epoch_duration:.2f}s")
        
        # Update running results file
        with open(results_file, "a") as f:
            f.write(f"{epoch}\t{avg_train_loss:.4f}\t{val_f1:.4f}\t{val_accuracy:.4f}\t{val_far:.4f}\t{epoch_duration:.2f}s\n")
            
        # --- CHECKPOINT & EARLY STOPPING ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            
            with open(results_file, "a") as f:
                f.write(f"--- New Best Model Saved at Epoch {epoch} with F1: {best_val_f1:.4f} ---\n")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= args.patience and epoch >= args.min_epochs:
            print(f"Early stopping triggered at epoch {epoch}.")
            with open(results_file, "a") as f:
                f.write(f"Early stopping triggered at epoch {epoch}.\n")
            break

    # Final wrap-up output
    with open(results_file, "a") as f:
        f.write(f"\nTraining Finished. Best Epoch: {best_epoch}, Best Val F1: {best_val_f1:.4f}\n")
        
    print(f"Training Finished. Best Val F1: {best_val_f1:.4f}")

    print("Per-class FAR breakdown:")
    for c in sorted(per_class_far.keys()):
        print(f"  class {c:3d}: FAR = {per_class_far[c]:.4f}")
    print(f"Mean FAR: {val_far:.4f}")

if __name__ == "__main__":
    train()