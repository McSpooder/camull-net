import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



class ConvBlock(nn.Module):

    def __init__(self, c_in, c_out, ks, k_stride=1):

        super().__init__()

        self.conv1 = nn.Conv3d(c_in, c_out, ks, stride=k_stride, padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(c_out)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2)
        self.dropout = nn.Dropout3d(p=0.1)

        
    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn(out)
        out = self.elu(out)
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
        self.sig  = nn.Sigmoid()


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
        out       = self.sig(out)
        
        return out

class ImprovedCamull(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Improved MRI processing branch
        self.mri_encoder = nn.ModuleList([
            # Initial conv with smaller kernel and no stride
            ConvBlock(1, 32, (3,3,3), k_stride=1),
            ConvBlock(32, 64, (3,3,3)),
            ConvBlock(64, 96, (3,3,3))
        ])
        
        # Residual blocks for better feature extraction
        self.res_blocks = nn.ModuleList([
            ResidualBlock(96),
            ResidualBlock(96),
            ResidualBlock(96)
        ])
        
        # Clinical data processing with attention
        self.clinical_encoder = nn.Sequential(
            FCBlock(21, 32),
            FCBlock(32, 32)
        )
        
        # Attention mechanism for better feature fusion
        self.attention = MultiModalAttention(96, 32)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            FCBlock(96 + 32, 64),
            FCBlock(64, 32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        mri, clinical = x
        
        # MRI processing with residual connections
        mri_features = mri
        skip_connections = []
        
        # Encoder path with skip connections
        for encoder in self.mri_encoder:
            mri_features = encoder(mri_features)
            skip_connections.append(mri_features)
        
        # Residual blocks
        for res_block in self.res_blocks:
            mri_features = res_block(mri_features)
        
        # Process clinical data
        clinical_features = self.clinical_encoder(clinical)
        
        # Attention-based fusion
        fused_features = self.attention(mri_features, clinical_features)
        
        # Final classification
        out = self.classifier(fused_features)
        
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
       
def load_cam_model(path):
    model = torch.load(path)
    return model
