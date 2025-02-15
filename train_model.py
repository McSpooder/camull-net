'''The following module trains the weights of the neural network model.'''

import glob
from data_declaration import Task
from architecture     import load_cam_model, Camull, ImprovedCamull
import torch
import torch.nn    as nn
import torch.optim as optim

from collections import Counter

from   sklearn.metrics   import roc_auc_score

import enlighten
from tqdm.auto import tqdm

import copy
import os
import sys
import datetime
import uuid

import numpy as np

_manager = enlighten.get_manager()  # Single manager instance

DEVICE = None


def start(device, ld_helper, epochs, model_uuid=None):
        
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
        '''The following function saves the weights file into required folder'''
        if sys.platform.__str__() == 'linux':
            root_path = "../weights/" + task.__str__() + "/" + uuid_arg + "/"
        else: #windows
            root_path = "..\\weights\\" + task.__str__() + "\\" + uuid_arg + "\\"

        if fold == 1: 
            os.makedirs(root_path, exist_ok=True)

        while True:
            s_path = root_path + "fold_{}_weights-{date:%Y-%m-%d_%H-%M-%S}.pt".format(
                fold, date=datetime.datetime.now()
            )

            if os.path.exists(s_path):
                print("Path exists. Choosing another path.")
            else:
                # Save only the state dict instead of the whole model
                torch.save(model_in.state_dict(), s_path)
                break


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

                if model_cop is None:
                    model = build_arch()
                else:
                    model = model_cop
                
                train_dl = ld_helper.get_train_dl(k_ind)
                val_dl = ld_helper.get_val_dl(k_ind)
                
                print(f"Starting train_loop with epochs={epochs}")
                fold_history = train_loop(model, train_dl, val_dl, epochs)
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

    def train_loop(model, train_dl, val_dl, epochs):
        '''Enhanced training loop with validation and metrics tracking'''
        # At the start, print model and data types
        for name, param in model.named_parameters():
            print(f"Parameter {name}: {param.dtype}")
        # At the start of train_loop
        # Add at the start of your train_loop function
        log_file = open('training_log.txt', 'w')
            # Then modify your print statements to also write to the file
        def log_print(message, console=True):
            """Log message to file and optionally to console"""
            log_file.write(message + '\n')
            log_file.flush()  # Ensure it's written immediately
            if console:
                print(message)

        train_label_dist = Counter([l.item() for batch in train_dl for l in batch['label']])
        val_label_dist = Counter([l.item() for batch in val_dl for l in batch['label']])

        for batch in train_dl:
            # Debug first batch
            print("\nBatch structure:")
            for k, v in batch.items():
                print(f"{k}: {v.dtype}, shape: {v.shape}")
            break
        epochs_c.total = epochs
        batches_c.total = len(train_dl)
        
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-5)
        # Convert optimizer state to float32
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.float()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        loss_function = nn.BCEWithLogitsLoss()

        history = {
            'train_loss': [], 'val_loss': [],
            'train_auc': [], 'val_auc': [],
            'best_val_auc': 0,
            'best_epoch': 0
        }
        
        for epoch in range(epochs):
            log_print(f"\nEpoch: {epoch+1}/{epochs}")
            # Training phase
            model.train()
            train_losses = []
            train_preds = []
            train_labels = []
            
            for batch_idx, sample_batched in enumerate(train_dl):
                try:
                    batch_x = sample_batched['mri'].to(DEVICE)
                    batch_xb = sample_batched['clin_t'].to(DEVICE)
                    batch_y = sample_batched['label'].to(DEVICE)
                    
                    # Forward pass
                    model.zero_grad()
                    outputs = model((batch_x, batch_xb))

                    if batch_idx == 0:
                        probs = torch.sigmoid(outputs)
                        log_print("\nFirst batch predictions:", console=False)  # Log to file only
                        log_print(f"Probability range: [{probs.min().item():.4f}, {probs.max().item():.4f}]", console=False)
                        for i in range(min(4, len(outputs))):
                            log_print(f"Sample {i}: Prob = {probs[i].item():.4f}, Label = {batch_y[i].item()}", console=False)

                    # Check for NaN values
                    if torch.isnan(outputs).any():
                        print(f"NaN detected in outputs at epoch {epoch}, batch {batch_idx}")
                        continue
                    
                    loss = loss_function(outputs, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Track metrics
                    train_losses.append(loss.item())
                    train_preds.extend(outputs.detach().cpu().numpy())
                    train_labels.extend(batch_y.cpu().numpy())
                    
                    batches_c.update()
                    
                except Exception as e:
                    print(f"Error in training batch {batch_idx}: {str(e)}")
                    continue
            
            # Validation phase
            model.eval()
            val_losses = []
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch_idx, sample_batched in enumerate(val_dl):
                    try:
                        batch_x = sample_batched['mri'].to(DEVICE)
                        batch_xb = sample_batched['clin_t'].to(DEVICE)
                        batch_y = sample_batched['label'].to(DEVICE)
                        
                        model.zero_grad()
                        outputs = model((batch_x, batch_xb))
                        
                        if torch.isnan(outputs).any():
                            print(f"NaN detected in validation outputs at epoch {epoch}, batch {batch_idx}")
                            continue
                            
                        loss = loss_function(outputs, batch_y)
                        
                        val_losses.append(loss.item())
                        val_preds.extend(outputs.cpu().numpy())
                        val_labels.extend(batch_y.cpu().numpy())
                        
                    except Exception as e:
                        print(f"Error in validation batch {batch_idx}: {str(e)}")
                        continue
            
            try:
                # Calculate epoch metrics
                if train_losses:
                    train_loss = np.mean(train_losses)
                else:
                    train_loss = float('nan')
                    
                if val_losses:
                    val_loss = np.mean(val_losses)
                else:
                    val_loss = float('nan')
                
                # Print raw values for debugging
                print(f"Train predictions shape: {np.array(train_preds).shape}")
                print(f"Train labels shape: {np.array(train_labels).shape}")
                print(f"Sample of predictions: {train_preds[:5]}")
                print(f"Sample of labels: {train_labels[:5]}")
                
                # Only calculate AUC if we have valid predictions and labels
                if len(train_preds) > 0 and len(train_labels) > 0:
                    train_preds_np = np.array(train_preds)
                    train_labels_np = np.array(train_labels)
                    
                    # Check for NaN or infinite values
                    if not np.any(np.isnan(train_preds_np)) and not np.any(np.isnan(train_labels_np)):
                        train_auc = roc_auc_score(train_labels_np, train_preds_np)
                    else:
                        train_auc = float('nan')
                else:
                    train_auc = float('nan')
                    
                if len(val_preds) > 0 and len(val_labels) > 0:
                    val_preds_np = np.array(val_preds)
                    val_labels_np = np.array(val_labels)
                    
                    if not np.any(np.isnan(val_preds_np)) and not np.any(np.isnan(val_labels_np)):
                        val_auc = roc_auc_score(val_labels_np, val_preds_np)
                    else:
                        val_auc = float('nan')
                else:
                    val_auc = float('nan')
                
                # Update history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_auc'].append(train_auc)
                history['val_auc'].append(val_auc)
                                
                 # Track best model
                if not np.isnan(val_auc) and val_auc > history['best_val_auc']:
                    history['best_val_auc'] = val_auc
                    history['best_epoch'] = epoch
                
                # Add scheduler step here, using validation AUC as the metric
                scheduler.step(val_auc)  # This line was missing
                # End of epoch metrics - show these in both console and file
                log_print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
                log_print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
                log_print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

                if (epoch + 1) % 5 == 0:
                    torch.save(model.state_dict(), f'model_checkpoint_epoch_{epoch+1}.pth')
                
            except Exception as e:
                print(f"Error calculating metrics: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            
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


