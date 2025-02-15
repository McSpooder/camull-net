import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import logging

def safe_std(tensor):
    """Safely calculate standard deviation for any tensor"""
    if tensor.numel() > 1:
        return tensor.std(unbiased=False)
    return torch.tensor(0.0, device=tensor.device)

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, ks, k_stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(c_in, c_out, ks, stride=k_stride, padding='same')
        self.bn = nn.BatchNorm3d(c_out)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout3d(p=0.1)
        
        # Convert all parameters to float32
        for param in self.parameters():
            param.data = param.data.float()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.dropout(out)
        return out



class ConvBlock2(nn.Module):
    
    def __init__(self, chans, ks, k_stride=1):
        
        super().__init__()
        conv1_ks = [1] + ks[:-1]
        conv2_ks = [ks[-1]] + [1,1]
        
        #depthwise
        self.conv1 = nn.Conv3d(chans, chans, kernel_size=conv1_ks, stride=k_stride, padding=(0,1,1), groups=chans)
        #pointwise
        self.conv2 = nn.Conv3d(chans, chans, kernel_size=conv2_ks, stride=k_stride, padding=(1,0,0))
        
        self.bn = nn.BatchNorm3d(chans)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout3d(p=0.1)
        

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.elu(out)
        out = self.dropout(out)
        
        return out
    


class FCBlock(nn.Module):
    def __init__(self, chan_in, units_out):
        super().__init__()
        self.fc = nn.Linear(chan_in, units_out)
        self.bn = nn.BatchNorm1d(units_out)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.1)
        
        # Convert all parameters to float32
        for param in self.parameters():
            param.data = param.data.float()

    def forward(self, x):
        
        out = self.fc(x)
        out = self.bn(out)
        out = self.elu(out)
        out = self.dropout(out)
        return out 
    


class Camull(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.stack1 = nn.Sequential(ConvBlock(1, 24, (11,11,11), 2),
                                   ConvBlock(24, 48, (5,5,5)))

        self.stack1_b = nn.Sequential(ConvBlock(1, 24, (11,11,11), 2),
                                     ConvBlock(24, 48, (5,5,5)))

        #seperable convolutions
        self.stack2  = nn.Sequential(ConvBlock2(96, [3,3,3]),
                                   ConvBlock2(96, [3,3,3]),
                                   ConvBlock2(96, [3,3,3]))

        #Number of channels inputted is halfed to reduce number of parameters.
        #This is done to the input in the forward function.
        self.stack3_a = nn.Sequential(ConvBlock(48, 24, (3,3,3)),
                                     ConvBlock(24, 8, (3,3,3)))

        self.stack3_b = nn.Sequential(ConvBlock(48, 24, (3,3,3)),
                                     ConvBlock(24, 8, (3,3,3)))

        self.fcblock = nn.Sequential(FCBlock(21, 32),
                                     FCBlock(32, 10))

        self.flat = nn.Flatten()
        self.fc1  = FCBlock(128, 10)
        self.fc2  = FCBlock(20, 4)
        self.lin  = nn.Linear(4, 1)


    #Performing a grouped convolutional stack
    def s3_forward(self, x):
        
        bound     = int(np.floor(x.shape[1]/2))
        out_a     = x[:,:bound]
        out_b     = x[:,bound:]  
            

        out_a     = self.stack3_a(out_a)
        out_b     = self.stack3_b(out_b)
        out       = torch.cat((out_a, out_b), 1)
        
        return out
        
        
    def cat_with_clin(self, x_a, x_b):
        
        out       = self.flat(x_a)       
        out_a     = self.fc1(out)
        out_b     = self.fcblock(x_b)
        out       = torch.cat((out_a, out_b), 1)
        
        return out
        
        
    def forward(self, x):
        
        mri, clin = x

        out_a     = self.stack1(mri)
        out_b     = self.stack1_b(mri)
        out       = torch.cat((out_a, out_b), 1) #1 as ind 0 is batch size    
        

        identity  = out
        out       = self.stack2(out)
        out       = out + identity
        

        out       = self.s3_forward(out)
        out       = self.cat_with_clin(out, clin)


        out       = self.fc2(out)
        out       = self.lin(out)
        
        return out

class ImprovedCamull(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Calculate the feature size after convolutions
        self.input_shape = (110, 110, 110)
        
        # MRI processing branch
        self.mri_encoder = nn.ModuleList([
            ConvBlock(1, 32, (3,3,3), k_stride=1).float(),
            ConvBlock(32, 64, (3,3,3)).float(),
            ConvBlock(64, 96, (3,3,3)).float()
        ])
        
        # Calculate the size after convolutions
        self._calculate_conv_output_size()
        
        # Clinical data processing
        self.clinical_encoder = nn.Sequential(
            FCBlock(21, 32).float(),
            FCBlock(32, 32).float()
        )
        
        # Add a dimension reduction layer before fusion
        self.dim_reduction = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conv_output_size, 256).float(),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 64).float(),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32).float(),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1).float(),
            nn.Tanh()
        )
        for param in self.parameters():
            param.data = param.data.float()
        self._force_float32()
        self._init_weights()

    def _force_float32(self):
            def convert_to_float32(module):
                for child in module.children():
                    convert_to_float32(child)
                
                for param in module._parameters:
                    if module._parameters[param] is not None:
                        module._parameters[param] = module._parameters[param].float()
                
                for buffer in module._buffers:
                    if module._buffers[buffer] is not None:
                        module._buffers[buffer] = module._buffers[buffer].float()

            convert_to_float32(self)
                
    def log_tensor_stats(self, tensor, name):
        if tensor.numel() > 1:  # Only calculate std if we have more than one element
            logging.info(f"{name} - Mean: {tensor.mean():.4f}, Std: {tensor.std():.4f}")
        else:
            logging.info(f"{name} - Mean: {tensor.mean():.4f}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                # Initialize with float32
                m.weight.data = m.weight.data.float()
                if m.bias is not None:
                    m.bias.data = m.bias.data.float()

    def _calculate_conv_output_size(self):
        # Helper function to calculate output size after convolutions
        x = torch.zeros(1, 1, *self.input_shape)
        
        # Pass through convolution layers
        for conv in self.mri_encoder:
            x = conv(x)
        
        self.conv_output_size = x.numel() // x.size(0)
        print(f"Convolution output size: {self.conv_output_size}")
    

    def forward(self, x):
        mri, clinical = x
        
        # Process MRI data through encoders
        mri_features = mri
        for i, encoder in enumerate(self.mri_encoder):
            mri_features = encoder(mri_features)
            logging.info(f"\n{'='*50}")
            logging.info(f"NETWORK LAYER STATISTICS - ENCODER {i+1}")
            self.log_tensor_stats(mri_features, "Values")  # Using the new method
            logging.info(f"Min: {mri_features.min():.4f}")
            logging.info(f"Max: {mri_features.max():.4f}")
        
        # After dimension reduction
        mri_features = self.dim_reduction(mri_features)
        logging.info(f"\n{'='*50}")
        logging.info(f"AFTER DIMENSION REDUCTION")
        self.log_tensor_stats(mri_features, "Values")
        
        # Clinical features
        clinical_features = self.clinical_encoder(clinical)
        logging.info(f"\n{'='*50}")
        logging.info(f"CLINICAL FEATURES")
        logging.info(f"Mean: {clinical_features.mean():.4f}")
        logging.info(f"Std: {safe_std(clinical_features):.4f}")
        
        # Combined features
        combined = torch.cat([mri_features, clinical_features], dim=1)
        logging.info(f"\n{'='*50}")
        logging.info(f"COMBINED FEATURES")
        logging.info(f"Mean: {combined.mean():.4f}")
        logging.info(f"Std: {safe_std(combined):.4f}")
        
        # Final classification
        out = self.classifier(combined)
        logging.info(f"\n{'='*50}")
        logging.info(f"FINAL OUTPUT")
        logging.info(f"Mean: {out.mean():.4f}")
        logging.info(f"Std: {safe_std(out):.4f}")
        
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout3d(0.1)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.elu(out)
        return out

class MultiModalAttention(nn.Module):
    def __init__(self, mri_channels, clinical_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(mri_channels + clinical_channels, 64),
            nn.ReLU(),
            nn.Linear(64, mri_channels),
            nn.Sigmoid()
        )
        
    def forward(self, mri_features, clinical_features):
        # Flatten MRI features for attention
        b, c, h, w, d = mri_features.shape
        mri_flat = mri_features.view(b, c, -1).mean(-1)
        
        # Compute attention weights
        combined = torch.cat([mri_flat, clinical_features], dim=1)
        weights = self.attention(combined).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # Apply attention and concatenate features
        attended_mri = (mri_features * weights).view(b, -1)
        return torch.cat([attended_mri, clinical_features], dim=1)
       
def load_cam_model(path, device):
    # First create a new instance of the model
    model = ImprovedCamull()
    
    # Load the state dict
    try:
        # First try loading with weights_only=True
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except:
        # If that fails, load without weights_only
        state_dict = torch.load(path, map_location=device)
        # If we loaded the whole model, get its state dict
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
    
    # Load the state dict into the model
    model.load_state_dict(state_dict)
    return model