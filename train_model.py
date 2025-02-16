'''The following module trains the weights of the neural network model.'''
import traceback
import glob
from data_declaration import Task
from architecture     import load_cam_model, Camull, ImprovedCamull
import torch
import torch.nn    as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from collections import Counter

from   sklearn.metrics   import roc_auc_score

import enlighten
import logging
from tqdm.auto import tqdm

import copy
import os
import sys
import datetime
import uuid

import numpy as np

_manager = enlighten.get_manager()  # Single manager instance

DEVICE = None

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_val_auc = None
        self.early_stop = False

    def __call__(self, val_auc):
        if self.best_val_auc is None:
            self.best_val_auc = val_auc
        elif val_auc < self.best_val_auc + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_auc = val_auc
            self.counter = 0

def start(device, ld_helper, epochs, model_uuid=None):
    
    def smooth_labels(labels, smoothing=0.1):
        """Apply label smoothing to binary labels"""
        return labels * (1 - smoothing) + smoothing * 0.5

    def log_metrics(train_loss, train_auc, val_loss, val_auc, lr, uuid, epoch, fold):
        """Log training metrics to a separate file with model identification"""
        with open('training_metrics.log', 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Model UUID: {uuid}\n")
            f.write(f"Fold: {fold}, Epoch: {epoch}\n")
            f.write(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}\n")
            f.write(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}\n")
            f.write(f"Learning Rate: {lr:.6f}\n")

    def load_model(arch, model_uuid=None):
        '''Function for loaded camull net from a specified weights path'''
        if arch == "camull": #must be camull

            if model_uuid is None:
                model = load_cam_model("../weights/camnet/fold_0_weights-2020-04-09_18_29_02", device)
            else:
                paths = glob.glob("../weights/NC_v_AD/{}/*".format(model_uuid))              
                model = load_cam_model(paths[0], device)

        return model

    def save_weights(model_in, uuid_arg, fold=1, task: Task = None):
        try:
            if sys.platform.__str__() == 'linux':
                root_path = "../weights/" + task.__str__() + "/" + uuid_arg + "/"
            else: #windows
                root_path = "..\\weights\\" + task.__str__() + "\\" + uuid_arg + "\\"

            if fold == 1: 
                os.makedirs(root_path, exist_ok=True)

            s_path = root_path + f"fold_{fold}_weights-{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.pt"
            
            # Set model to eval mode before saving
            model_in.eval()
            torch.save(model_in.state_dict(), s_path)
            # Return to training mode if needed
            model_in.train()
            
            print(f"Successfully saved weights to {s_path}")

        except Exception as e:
            print(f"Error saving weights: {str(e)}")
            traceback.print_exc()
            raise


    def build_arch():
        '''Function for instantiating the pytorch neural network object'''
        net = ImprovedCamull()
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
        
        net.to(DEVICE)
        net.float()
        
        return net

    def train_camull(ld_helper, k_folds=5, model=None, epochs=40):
        print(f"\nEntering train_camull function")
        print(f"Current module: {__name__}")
        try:
            print(f"Type of epochs: {type(epochs)}, Value: {epochs}")
            print(f"Starting training with {k_folds} folds")
            
            folds_c.total = k_folds
            task = ld_helper.get_task()
            uuid_ = uuid.uuid4().hex
            model_cop = model
            
            # Track metrics across folds
            fold_metrics = []
            best_val_auc = 0
            best_model_state = None

            print("\nOverall Dataset Statistics:")
            ld_helper.print_dataset_stats()
            
            for k_ind in range(k_folds):
                print(f"\n=========== Training on Fold {k_ind + 1}/{k_folds} ===========")
                ld_helper.print_fold_stats(k_ind)
                ld_helper.print_split_stats(k_ind)
                if model_cop is None:
                    model = build_arch()
                else:
                    model = model_cop
                
                train_dl = ld_helper.get_train_dl(k_ind)
                val_dl = ld_helper.get_val_dl(k_ind)
                
                print(f"Starting train_loop with epochs={epochs}")
                fold_history = train_loop(
                    model, 
                    train_dl, 
                    val_dl, 
                    epochs=epochs,
                    uuid=uuid_,
                    fold=k_ind+1
                )
                fold_metrics.append(fold_history)
                
                # Save if best model
                if fold_history['best_val_auc'] > best_val_auc:
                    best_val_auc = fold_history['best_val_auc']
                    best_model_state = copy.deepcopy(model.state_dict())
                
                # Save fold weights
                save_weights(model, uuid_, fold=k_ind+1, task=task)
                folds_c.update()

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
        
        # Save best model across all folds
        if best_model_state is not None:
            torch.save(best_model_state, f'../models/best_model_{uuid_}.pth')
            
        # Print final cross-validation results
        print_cv_results(fold_metrics)
        
        folds_c.count = 0
        return uuid_

    def setup_training(model, train_dl, epochs):
        """Initialize training components"""
        # Setup logging
        log_file = open('training_log.txt', 'w')
        
        # Setup data counters
        epochs_c.total = epochs
        batches_c.total = len(train_dl)
        
        # Calculate class weights
        train_label_dist = Counter([l.item() for batch in train_dl for l in batch['label']])
        pos_weight = torch.tensor([train_label_dist[1] / train_label_dist[0]]).to(DEVICE)
        
        # Setup optimizer and loss
        optimizer = setup_optimizer(model)

        loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        return log_file, optimizer, loss_function

    def initialize_history():
        """Initialize dictionary to track training history"""
        return {
            'train_loss': [], 'val_loss': [],
            'train_auc': [], 'val_auc': [],
            'best_val_auc': 0,
            'best_epoch': 0,
            'learning_rates': []  # Add this
        }

    def setup_optimizer(model):
        """Setup optimizer with proper settings"""
        optimizer = optim.AdamW(  # Use AdamW
            model.parameters(),
            lr=0.0001,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.float()
        return optimizer
    
    def update_metrics(history, train_losses, train_preds, train_labels, 
                  val_losses, val_preds, val_labels, optimizer):
        """Calculate and update metrics for both training and validation"""
        metrics = {}
        
        # Calculate training metrics
        if train_losses:
            metrics['train_loss'] = np.mean(train_losses)
            if len(train_preds) > 0 and len(train_labels) > 0:
                train_preds_np = np.array(train_preds)
                train_labels_np = np.array(train_labels)
                if not np.any(np.isnan(train_preds_np)) and not np.any(np.isnan(train_labels_np)):
                    metrics['train_auc'] = roc_auc_score(train_labels_np, train_preds_np)
                else:
                    metrics['train_auc'] = float('nan')
        else:
            metrics['train_loss'] = float('nan')
            metrics['train_auc'] = float('nan')
        
        # Calculate validation metrics
        if val_losses:
            metrics['val_loss'] = np.mean(val_losses)
            if len(val_preds) > 0 and len(val_labels) > 0:
                val_preds_np = np.array(val_preds)
                val_labels_np = np.array(val_labels)
                if not np.any(np.isnan(val_preds_np)) and not np.any(np.isnan(val_labels_np)):
                    metrics['val_auc'] = roc_auc_score(val_labels_np, val_preds_np)
                else:
                    metrics['val_auc'] = float('nan')
        else:
            metrics['val_loss'] = float('nan')
            metrics['val_auc'] = float('nan')
        
        # Update history
        for key in ['train_loss', 'val_loss', 'train_auc', 'val_auc']:
            history[key].append(metrics[key])

        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        metrics['lr'] = current_lr
        # Print metrics
        print(f"Train Loss: {metrics['train_loss']:.4f}, Train AUC: {metrics['train_auc']:.4f}")
        print(f"Val Loss: {metrics['val_loss']:.4f}, Val AUC: {metrics['val_auc']:.4f}")
        return metrics
    
    def save_checkpoint(model, metrics, epoch, history, fold):
        """Save model checkpoint if it's the best so far"""
        if metrics['val_auc'] > history['best_val_auc']:
            history['best_val_auc'] = metrics['val_auc']
            history['best_epoch'] = epoch
            torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')
        
        # Save regular checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'model_checkpoint_epoch_{epoch+1}.pth')
    
    def train_epoch(model, train_dl, optimizer, loss_function, epoch):
        """Run one epoch of training"""
        model.train()
        train_losses, train_preds, train_labels = [], [], []
        
        for batch_idx, sample_batched in enumerate(train_dl):
            try:
                batch_x = sample_batched['mri'].to(DEVICE)
                batch_xb = sample_batched['clin_t'].to(DEVICE)
                batch_y = sample_batched['label'].to(DEVICE)
                
                # Apply label smoothing
                smoothed_y = smooth_labels(batch_y)
                
                model.zero_grad()
                outputs = model((batch_x, batch_xb))
                loss = loss_function(outputs, smoothed_y)  # Use smoothed labels
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Using 1.0 as clip value

                optimizer.step()
                
                train_losses.append(loss.item())
                train_preds.extend(outputs.detach().cpu().numpy())
                train_labels.extend(batch_y.cpu().numpy())  # Keep original labels for metrics
                
                batches_c.update()
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {str(e)}")
                continue
                
        return train_losses, train_preds, train_labels
    
    def validate(model, val_dl, loss_function):
        """Run validation"""
        model.eval()
        val_losses, val_preds, val_labels = [], [], []
        
        with torch.no_grad():
            for batch_idx, sample_batched in enumerate(val_dl):
                try:
                    loss, outputs, batch_y = process_batch(
                        model, sample_batched, loss_function, None, is_training=False
                    )
                    val_losses.append(loss.item())
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(batch_y.cpu().numpy())
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue
                    
        return val_losses, val_preds, val_labels
    
    def process_batch(model, batch, loss_function, optimizer=None, is_training=True):
        """Process a single batch"""
        batch_x = batch['mri'].to(DEVICE)
        batch_xb = batch['clin_t'].to(DEVICE)
        batch_y = batch['label'].to(DEVICE)
        
        if is_training:
            model.zero_grad()
        
        outputs = model((batch_x, batch_xb))
        loss = loss_function(outputs, batch_y)
        
        if is_training and optimizer is not None:
            loss.backward()
            optimizer.step()
        
        return loss, outputs, batch_y

    def train_loop(model, train_dl, val_dl, epochs=40, uuid=None, fold=None):
        """Main training loop"""
        log_file, optimizer, loss_function = setup_training(model, train_dl, epochs)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        early_stopping = EarlyStopping(patience=5)
        history = initialize_history()
        
        for epoch in range(epochs):
            print(f"\nEpoch: {epoch+1}/{epochs}")
            
            # Training phase
            train_losses, train_preds, train_labels = train_epoch(
                model, train_dl, optimizer, loss_function, epoch
            )
            
            # Validation phase
            val_losses, val_preds, val_labels = validate(model, val_dl, loss_function)
            
            # Update metrics and check early stopping
            metrics = update_metrics(
                history, train_losses, train_preds, train_labels,
                val_losses, val_preds, val_labels, optimizer
            )
            
            log_metrics(
                train_loss=metrics['train_loss'],
                train_auc=metrics['train_auc'],
                val_loss=metrics['val_loss'],
                val_auc=metrics['val_auc'],
                lr=optimizer.param_groups[0]['lr'],
                uuid=uuid,  # Add this
                epoch=epoch,  # Add this
                fold=fold  # Add this
            )
            
            early_stopping(metrics['val_auc'])
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            # Save best model and update scheduler
            save_checkpoint(model, metrics, epoch, history, folds_c.count)
            scheduler.step(metrics['val_auc'])
            
            # Reset batch counter
            batches_c.count = 0
            epochs_c.update()
        
        log_file.close()
        epochs_c.count = 0
        return history

    def print_cv_results(fold_metrics):
        '''Print cross-validation results with error handling'''
        try:
            val_aucs = [m['best_val_auc'] for m in fold_metrics if not np.isnan(m['best_val_auc'])]
            if val_aucs:  # Only calculate if we have valid AUCs
                mean_auc = np.mean(val_aucs)
                std_auc = np.std(val_aucs)
                
                print("\nCross-validation Results:")
                print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
                for i, metrics in enumerate(fold_metrics):
                    print(f"Fold {i+1} Best AUC: {metrics['best_val_auc']:.4f} "
                        f"(Epoch {metrics['best_epoch']+1})")
            else:
                print("\nNo valid AUC scores to report")
                
        except Exception as e:
            print(f"Error in print_cv_results: {str(e)}")
            import traceback
            traceback.print_exc()
        
    try:
        #manager = enlighten.get_manager()
        folds_c = _manager.counter(total=5, desc='Fold', unit='folds')
        epochs_c = _manager.counter(total=epochs, desc='Epochs', unit='epochs')
        batches_c = _manager.counter(total=75, desc='Batches', unit='batches')

        task = ld_helper.get_task()
        DEVICE = device
        
        if (task == Task.NC_v_AD):
            model_uuid = train_camull(ld_helper=ld_helper, epochs=epochs)
        else: # sMCI vs pMCI
            if (model_uuid != None):
                model = load_model("camull", model_uuid)
                model_uuid = train_camull(ld_helper, model=model, epochs=epochs)
            else:
                print("Need a model uuid")
                return None
        
        return model_uuid
        
    except Exception as e:
        print(f"Error in start(): {str(e)}")
        return None


