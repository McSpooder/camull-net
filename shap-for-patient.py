import joblib
import numpy as np
import shap
import torch 
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    
shap.initjs()

# Load the explainer from the file
explainer = joblib.load('C:\\Users\\Hextr\\Programming\\alzheimers\\camull-net\\shap_explainer.pkl')
random_data = np.random.randn(1, 21).astype(np.float64)
random_data = torch.from_numpy(random_data)
random_data = random_data.to(device)

clinical_data = np.random.randn(1, 21).astype(np.float64)

# Now you can use the loaded explainer to get SHAP values
shap_values = explainer.shap_values(random_data)
shap_values = shap_values.squeeze(-1)  # Shape now becomes (1, 21)

feature_names = [
    "AGE", "PTEDUCAT", "APOE4", "CDRSB", "ADAS11", "ADAS13", 
    "RAVLT_immediate", "RAVLT_learning", "RAVLT_forgetting", "RAVLT_perc_forgetting",
    "GENDER", "ETHNICITY_Hisp/Latino", "ETHNICITY_Not_Hisp/Latino", "ETHNICITY_Unknown",
    "RACCAT_Am_Indian/Alaskan", "RACCAT_Asian", "RACCAT_Black", "RACCAT_Hawaiian/Other_PI",
    "RACCAT_More_than_one", "RACCAT_Unknown", "RACCAT_White"
]

print(f"Type of shap_values: {type(shap_values)}")
print(f"Shape of shap_values: {np.shape(shap_values)}")
print(f"Expected Value: {explainer.expected_value}")



# Generate SHAP force plot for visualization
force_plot = shap.force_plot(explainer.expected_value, shap_values[0], feature_names)

# Save as an HTML file
shap.save_html("shap_force_plot.html", force_plot)

print ("shap_values: ", shap_values[0])
fig = shap.force_plot(explainer.expected_value, shap_values[0], feature_names,matplotlib=True)
force_plot = shap.plots.force(explainer.expected_value, shap_values[0], feature_names,matplotlib=True)
# Save as an HTML file
#force_plot.save_html("shap_force_plot.html")
plt.savefig("shap_force_plot.png", dpi=300, bbox_inches='tight')
plt.close(fig)
plt.show()