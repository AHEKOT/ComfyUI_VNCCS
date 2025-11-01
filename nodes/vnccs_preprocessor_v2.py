"""VNCCS Preprocessor V2 node

This node loads a custom preprocessor from a specified directory and applies it to an image.

- INPUTS: image, preprocessor_name
- OUTPUTS: preprocessed image

This file relies on runtime objects provided by ComfyUI.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import folder_paths


# Determine preprocessor path dynamically
def _get_preprocessor_path():
    """Get preprocessor path from environment or ComfyUI models directory"""
    # Check environment variable first
    if 'VNCCS_PREPROCESSOR_PATH' in os.environ:
        return os.environ['VNCCS_PREPROCESSOR_PATH']
    
    # Use ComfyUI's folder_paths to get correct models directory
    sams_path = os.path.join(folder_paths.models_dir, "sams")
    os.makedirs(sams_path, exist_ok=True)
    return sams_path


PREPROCESSOR_PATH = _get_preprocessor_path()
print(f"[VNCCS] Preprocessor path: {PREPROCESSOR_PATH}")


class PreprocessorModel(nn.Module):
    """
    Нейронная сеть для препроцессора (OpenPose, Canny, Depth и т.д.)
    Основана на архитектуре U-Net для задач image-to-image translation
    """
    
    def __init__(self, in_channels=3, out_channels=3, depth=64):
        super(PreprocessorModel, self).__init__()
        self.depth = depth
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, depth)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = self.conv_block(depth, depth * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = self.conv_block(depth * 2, depth * 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = self.conv_block(depth * 4, depth * 8)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(depth * 8, depth * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(depth * 16, depth * 8, 2, stride=2)
        self.dec4 = self.conv_block(depth * 16, depth * 8)
        
        self.upconv3 = nn.ConvTranspose2d(depth * 8, depth * 4, 2, stride=2)
        self.dec3 = self.conv_block(depth * 8, depth * 4)
        
        self.upconv2 = nn.ConvTranspose2d(depth * 4, depth * 2, 2, stride=2)
        self.dec2 = self.conv_block(depth * 4, depth * 2)
        
        self.upconv1 = nn.ConvTranspose2d(depth * 2, depth, 2, stride=2)
        self.dec1 = self.conv_block(depth * 2, depth)
        
        self.final = nn.Conv2d(depth, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        output = self.final(dec1)
        return output


def _load_preprocessor(preprocessor_name):
    """Load a PyTorch preprocessor model from the PREPROCESSOR_PATH"""
    model_path = os.path.join(PREPROCESSOR_PATH, f"{preprocessor_name}.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Preprocessor not found: {model_path}")
    
    print(f"[VNCCS] Loading preprocessor from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract config if available, otherwise use defaults
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        model_state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        config = checkpoint.get('config', {})
        model_state_dict = checkpoint['model_state_dict']
    else:
        # Just state_dict
        config = {}
        model_state_dict = checkpoint
    
    in_channels = config.get('in_channels', 3)
    out_channels = config.get('out_channels', 3)
    depth = config.get('depth', 64)
    
    # Create model
    model = PreprocessorModel(in_channels=in_channels, out_channels=out_channels, depth=depth)
    
    # Load state dict
    model.load_state_dict(model_state_dict)
    model.eval()
    return model


def _get_available_preprocessors():
    """Get list of available .pth preprocessor files"""
    preprocessors = []
    if os.path.exists(PREPROCESSOR_PATH):
        for file in os.listdir(PREPROCESSOR_PATH):
            if file.endswith('.pth'):
                preprocessors.append(file[:-4])  # Remove .pth extension
    return sorted(preprocessors) if preprocessors else ["best_model"]


class VNCCS_PreprocessorV2:
    @classmethod
    def INPUT_TYPES(s):
        preprocessors = _get_available_preprocessors()
        return {
            "required": {
                "image": ("IMAGE", ),
                "preprocessor_name": (preprocessors, {"default": preprocessors[0] if preprocessors else "best_model"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_preprocessor"

    CATEGORY = "VNCCS/preprocessing"

    def apply_preprocessor(self, image, preprocessor_name, width):
        if len(image) > 1:
            raise Exception('[VNCCS] ERROR: VNCCS_PreprocessorV2 does not allow image batches.')

        print(f'[VNCCS] DEBUG: Loading preprocessor: {preprocessor_name}')
        print(f'[VNCCS] DEBUG: Input image shape: {image.shape}, dtype: {image.dtype}')
        
        try:
            preprocessor = _load_preprocessor(preprocessor_name)
        except Exception as e:
            raise Exception(f'[VNCCS] ERROR: Failed to load preprocessor: {str(e)}')

        print(f'[VNCCS] DEBUG: Preprocessor type: {type(preprocessor).__name__}')
        
        try:
            # Ensure input is float32 and in range [0, 1]
            input_tensor = image.clone().float()
            if input_tensor.max() > 1.0:
                input_tensor = input_tensor / 255.0
            
            # Ensure [B, C, H, W] format for model
            if len(input_tensor.shape) == 3:
                input_tensor = input_tensor.unsqueeze(0)
            
            # Convert from [B, H, W, C] to [B, C, H, W] if needed
            if input_tensor.shape[-1] in [1, 3, 4] and input_tensor.shape[1] > input_tensor.shape[-1]:
                input_tensor = input_tensor.permute(0, 3, 1, 2)
            
            # Resize to target width while maintaining aspect ratio
            batch_size, channels, height, current_width = input_tensor.shape
            scale = width / current_width
            new_height = int(height * scale)
            
            if (current_width, height) != (width, new_height):
                input_tensor = F.interpolate(input_tensor, size=(new_height, width), mode='bilinear', align_corners=False)
                print(f'[VNCCS] DEBUG: Resized from ({height}, {current_width}) to ({new_height}, {width})')
            
            print(f'[VNCCS] DEBUG: Preprocessor input shape: {input_tensor.shape}')
            
            # Call the model/preprocessor with the image
            with torch.no_grad():
                result = preprocessor(input_tensor)
            
            print(f'[VNCCS] DEBUG: Preprocessor output shape: {result.shape}, dtype: {result.dtype}')
            
            # Ensure result is a tensor
            if isinstance(result, torch.Tensor):
                # Clamp to [0, 1] range and convert to [B, H, W, C] format
                result = torch.clamp(result, 0, 1)
                
                if len(result.shape) == 4 and result.shape[1] in [1, 3, 4]:
                    result = result.permute(0, 2, 3, 1)
                
                print(f'[VNCCS] DEBUG: Final output shape: {result.shape}')
                return (result,)
            else:
                raise RuntimeError(f"Preprocessor returned unsupported type: {type(result)}")
                
        except Exception as e:
            raise Exception(f'[VNCCS] ERROR: Failed to apply preprocessor: {str(e)}')


NODE_CLASS_MAPPINGS = {
    "VNCCS_PreprocessorV2": VNCCS_PreprocessorV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VNCCS_PreprocessorV2": "VNCCS Preprocessor V2",
}
