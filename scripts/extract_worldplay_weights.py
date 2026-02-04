#!/usr/bin/env python3
"""
Extract inference-ready weights from WorldPlay training checkpoint.
This script uses memory-efficient loading to handle the 49GB model.pt file.
"""

import torch
import os
import gc
import sys

def extract_generator_weights(input_path, output_path):
    print(f"Loading checkpoint from {input_path}...")
    print("This may take a while for a 49GB file...")
    
    # Use mmap for memory efficiency
    try:
        state_dict = torch.load(input_path, map_location='cpu', weights_only=True)
    except Exception as e:
        print(f"weights_only failed, trying standard load: {e}")
        state_dict = torch.load(input_path, map_location='cpu')
    
    print(f"Checkpoint keys: {list(state_dict.keys())}")
    
    # Extract generator
    if 'generator' not in state_dict:
        print("ERROR: No 'generator' key found!")
        print(f"Available keys: {list(state_dict.keys())}")
        sys.exit(1)
    
    generator_state = state_dict['generator']
    
    # Free memory from other parts
    del state_dict
    gc.collect()
    
    print(f"Generator has {len(generator_state)} keys")
    
    # Clean prefixes
    cleaned = {}
    for key, value in generator_state.items():
        new_key = key
        if new_key.startswith('model.'):
            new_key = new_key[6:]
        if new_key.startswith('_fsdp_wrapped_module.'):
            new_key = new_key[21:]
        cleaned[new_key] = value
    
    # Free original
    del generator_state
    gc.collect()
    
    print(f"Cleaned has {len(cleaned)} keys")
    print(f"Saving to {output_path}...")
    
    # Save as safetensors
    try:
        from safetensors.torch import save_file
        save_file(cleaned, output_path)
    except ImportError:
        # Fallback to PyTorch format
        output_path = output_path.replace('.safetensors', '.pt')
        torch.save(cleaned, output_path)
    
    size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"Done! Output size: {size_gb:.2f} GB")
    print(f"Saved to: {output_path}")
    
    return output_path

if __name__ == '__main__':
    input_path = r'D:\ComfyUI-Easy-Install\ComfyUI\models\worldplay\wan_distilled_model\model.pt'
    output_path = r'D:\ComfyUI-Easy-Install\ComfyUI\models\worldplay\wan_distilled_model\model_inference.safetensors'
    
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    extract_generator_weights(input_path, output_path)
