import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture gradients and activations
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_mri, target_class=None):
        """
        Generate Grad-CAM heatmap for an input MRI scan.
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_mri)
        print(output.shape)
        print("Output for class prediction:", output)
        print("target_class:", output.argmax().item())
        if target_class is None:
            target_class = output.argmax().item()

        # Backward pass
        # output[:, target_class].backward(retain_graph=True)

        output = output.squeeze()  # Remove extra dimension
        output.backward(retain_graph=True)  # Backpropagate on the score

        print("Gradients shape:", self.gradients.shape)
        # Compute Grad-CAM heatmap
        weights = self.gradients.mean(dim=[2, 3, 4], keepdim=True)  # Global average pooling
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU to remove negative values

        # Normalize to [0,1]
        cam -= cam.min()
        cam /= cam.max()

        return cam.cpu().numpy()
    
    def save_heatmap_as_nifti(self, heatmap, reference_nifti_path, output_path="gradcam_heatmap.nii.gz"):
        """
        Save the heatmap as a NIfTI (.nii.gz) file using a reference MRI scan for affine transformation.
        
        Args:
            heatmap (numpy.ndarray): The Grad-CAM heatmap (3D array).
            reference_nifti_path (str): Path to a reference MRI scan (.nii) to get affine and header info.
            output_path (str): Path to save the heatmap as a NIfTI file.
        """
        # Load reference MRI scan (to get spatial information)
        reference_nifti = nib.load(reference_nifti_path)
        
        # Ensure heatmap has the correct shape (remove singleton dimensions)
        heatmap = np.squeeze(heatmap)
        
        # Create a NIfTI image using the heatmap and reference affine
        heatmap_nifti = nib.Nifti1Image(heatmap, affine=reference_nifti.affine, header=reference_nifti.header)
        
        # Save the NIfTI file
        nib.save(heatmap_nifti, output_path)
        print(f"Grad-CAM heatmap saved to {output_path}")


