# model_loader.py
# This file is dedicated to loading the machine learning model.
# Separating this logic keeps the main app file clean.

import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def load_prediction_model(model_path="best_model-v3.pt"):
    """
    Loads the EfficientNet model.
    
    Args:
        model_path (str, optional): The path to your custom trained model weights (.pt file).
                                    If None, it will load the default ImageNet weights.
    
    Returns:
        A PyTorch model instance in evaluation mode.
    """
    # If you have a custom model file (like 'best_model-v3.pt' from your notebook),
    # place it in the same directory and provide the path.
    # For now, we use the pre-trained ImageNet weights as a placeholder.
    if model_path:
        print(f"Loading custom model from: {model_path}")
        # Initialize model with no pre-trained weights
        model = efficientnet_b0(weights=None)
        
        # The classifier head must match the one you trained with.
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(in_features, 2) # Assuming 2 classes: REAL and FAKE
        )
        
        # Load your custom weights
        # Use map_location to ensure it works on CPU if no GPU is available.
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Handle potential key mismatches (e.g., if saved from a LightningModule)
        if all(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
    else:
        print("No custom model path provided. Loading pre-trained EfficientNet_B0 on ImageNet.")
        # Load the model with standard pre-trained weights from ImageNet
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Set the model to evaluation mode. This is crucial for inference.
    model.eval()
    return model