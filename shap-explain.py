import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from camull_cli import load_cam_model
from data_declaration     import get_mri
from data_standardisation import convert_to_np

import torch
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda")

class ClinicalOnlyWrapper(torch.nn.Module):
    def __init__(self, model, mri_shape):
        """
        model: your original model that expects (mri, clinical_data)
        mri_shape: expected shape of an MRI input, e.g. (1, 110, 110, 110) without the batch dimension.
        """
        super(ClinicalOnlyWrapper, self).__init__()
        self.model = model
        self.mri_shape = mri_shape

    def forward(self, clin_input):
        print("forward Clinical Data Shape:", clin_input.shape)  
        print("forward Number of elements in Clinical Data:", clin_input.numel())  

        # Dynamically determine batch size
        batch_size = int(clin_input.shape[0])  # Explicitly convert to int

        # Create a baseline MRI input (here, zeros)
        baseline_mri = torch.zeros((batch_size, *self.mri_shape), 
                                device=clin_input.device, dtype=clin_input.dtype)

        # Create the MRI sample (assuming shape [1, 110, 110, 110])
        # mri_sample = np.random.rand(batch_size, 110, 110, 110).astype(np.float64)  # Multiple samples
        # mri_sample = torch.from_numpy(mri_sample)
        # mri_sample = mri_sample.to(device)
        # mri_sample = mri_sample.unsqueeze(1)  # Add channel dimension

        
        # Pass the tuple (mri_sample, clin_input) to the original model
        return self.model((baseline_mri, clin_input))

def get_data_sample():
    form_data = {}
    form_data['AGE'] = int(0)
    form_data['GENDER'] = "Female"
    form_data['ETHNICITY'] = "Hisp/Latino"
    form_data['RACCAT'] = "White"
    form_data['PTEDUCAT'] = int(0)
    form_data['APOE4'] = int(0)
    form_data['CDRSB'] = float(0)
    form_data['ADAS11'] = float(0)
    form_data['ADAS13'] = float(0)
    form_data['RAVLT_immediate'] = int(0)
    form_data['RAVLT_learning'] = int(0)
    form_data['RAVLT_forgetting'] = int(0)
    form_data['RAVLT_perc_forgetting'] = float(0)
    clin2 = convert_to_np(form_data)
    clin2 = torch.from_numpy(clin2)
    clin2 = clin2.to(device)
    return clin2  # Ensure it matches model expectation


model = load_cam_model("C:\\Users\\Hextr\\Programming\\alzheimers\\weights\\sMCI_v_pMCI\\5b5732a9904643b5b236cbabe6605dd7\\fold_1_weights-2020-05-05_10_29_43")
model.to(device)
model.eval()

#mri_sample = np.random.randn(1, 110, 110, 110) 
mri_sample = np.random.rand(1, 110, 110, 110).astype(np.float64)  # Uniformly between 0 and 1mri_sample = torch.from_numpy(mri_sample)
mri_sample = torch.from_numpy(mri_sample) 
mri_sample = mri_sample.unsqueeze(0)
mri_sample = mri_sample.to(device)
mri = get_mri("..\\inference\\ad-scan.nii")
mri = torch.from_numpy(mri) / 255.0 
mri = mri.unsqueeze(0)
mri = mri.to(device)
mri = (mri - mri.min()) / (mri.max() - mri.min() + 1e-8)  # Normalize to [0, 1]


print("Real MRI Stats - Min:", mri.min().item(), "Max:", mri.max().item(), "Mean:", mri.mean().item())
print("Random MRI Stats - Min:", mri_sample.min().item(), "Max:", mri_sample.max().item(), "Mean:", mri_sample.mean().item())
print(mri.shape)
#model_output = model((mri, get_data_sample()))

with torch.no_grad():
        net_out = model((mri_sample.view(-1, 1, 110, 110, 110), get_data_sample().view(1, 21)))
        net_out_real = model((mri.view(-1, 1, 110, 110, 110), get_data_sample().view(1, 21)))
print("Model output unwrapped random:", net_out)
print("Model output unwrapped real:", net_out_real)


if isinstance(model, torch.nn.DataParallel):
    model = model.module

mri_shape = (1, 110, 110, 110)
clinical_wrapper = ClinicalOnlyWrapper(model, mri_shape).to(device)
clinical_wrapper.eval()



# Generate a batch of sample data (Here I am assuming you have 3D MRI and clinical data)
# Replace this with actual data preprocessing or a specific batch
mri_sample = np.random.randn(2, 1, 110, 110, 110)  # Example: 2 samples, 1 channel, 64x64x64 volume


clinical_sample = get_data_sample()
clinical_sample = clinical_sample.to(device)

# Create SHAP input: in this case, just the clinical data
# Ensure it has a batch dimension (e.g., [1, num_features])
clinical_input = clinical_sample.unsqueeze(0)  # if clinical_sample was [num_features], now it's [1, num_features]
print("clinical_input shape:", clinical_input.shape)

random_data = np.random.randn(1, 21)
random_data = torch.from_numpy(random_data)
random_data = random_data.to(device)

model_output = clinical_wrapper(random_data)
print("Model output:", model_output)
print("Model output shape:", model_output.shape)

# Initialize SHAP DeepExplainer with the clinical wrapper and the clinical input.
import shap

random_data = np.random.randn(1, 21).astype(np.float64)
random_data = torch.from_numpy(random_data)
random_data = random_data.to(device)
clinical_batch = random_data
clinical_batch = np.random.randn(20, 21).astype(np.float64)
clinical_batch = torch.from_numpy(clinical_batch).to(device)
#random_data = get_data_sample()
#random_data = random_data.unsqueeze(0)
print("Clinical shape:", random_data.shape)  # Should be (1, 21)
explainer = shap.DeepExplainer(clinical_wrapper, clinical_batch)
joblib.dump(explainer, 'shap_explainer.pkl')
shap_values = explainer.shap_values(clinical_batch)
print(shap_values)

# If shap_values is a list (for a scalar model output, it often is a list with one element)
if isinstance(shap_values, list) and len(shap_values) == 1:
    shap_vals = shap_values[0]
else:
    shap_vals = shap_values

print("Raw SHAP values shape:", shap_vals.shape)

# If there is an extra dimension (e.g. shape is (1, 21, 1)), remove it:
if shap_vals.ndim == 3 and shap_vals.shape[-1] == 1:
    shap_vals = np.squeeze(shap_vals, axis=-1)

print("Squeezed SHAP values shape:", shap_vals.shape)
print("Clinical input shape:", clinical_input.detach().cpu().numpy().shape)

# Define the original feature names (before encoding)
feature_names = [
    "AGE", "PTEDUCAT", "APOE4", "CDRSB", "ADAS11", "ADAS13", 
    "RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting", "RAVLT_perc_forgetting",
    "GENDER", "ETHNICITY_Hisp/Latino", "ETHNICITY_Not_Hisp/Latino", "ETHNICITY_Unknown",
    "RACCAT_Am_Indian/Alaskan", "RACCAT_Asian", "RACCAT_Black", "RACCAT_Hawaiian/Other_PI",
    "RACCAT_More_than_one", "RACCAT_Unknown", "RACCAT_White"
]

# Ensure the number of feature names matches the number of SHAP values
assert len(feature_names) == clinical_input.shape[1], "Feature names mismatch!"

#feature_names = np.array([f"Feature_{i}" for i in range(clinical_input.shape[1])])
shap.summary_plot(shap_vals, clinical_batch.detach().cpu().numpy(), feature_names=feature_names)

# explainer = shap.DeepExplainer(clinical_wrapper, clinical_input)
# # Compute SHAP values
# shap_values = explainer.shap_values(clinical_input)
# print(shap_values)
# shap.summary_plot(shap_values, clinical_input.detach().cpu().numpy())

# # Now you can visualize SHAP values for the clinical data.
# shap.summary_plot(shap_values, clinical_input.detach().cpu().numpy(),
#                     feature_names = np.array([f"Feature_{i}" for i in range(21)]))