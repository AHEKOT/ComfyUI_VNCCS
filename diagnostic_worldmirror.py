
import torch
import sys
import os

# Add worldmirror to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WORLDMIRROR_DIR = os.path.join(PROJECT_ROOT, "background-data", "worldmirror")
if WORLDMIRROR_DIR not in sys.path:
    sys.path.insert(0, WORLDMIRROR_DIR)

from src.models.models.worldmirror import WorldMirror

def diagnostic():
    print("🔍 [Diagnostic] Loading WorldMirror model...")
    device = "cpu" # Test on CPU for stability
    
    try:
        model = WorldMirror.from_pretrained(
            "tencent/HunyuanWorld-Mirror"
        )
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully via from_pretrained")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Check some keys
    state_dict = model.state_dict()
    print(f"📊 State dict size: {len(state_dict)} keys")
    
    # Check for missing buffers
    buffers = dict(model.named_buffers())
    print(f"📊 Buffers: {list(buffers.keys())}")
    
    if "_resnet_mean" in buffers:
        print(f"   - resnet_mean: {buffers['_resnet_mean'].flatten()}")
    else:
        print("   ⚠️ _resnet_mean NOT in buffers (Matches persistent=False behavior)")

    # Check learnable tokens
    print(f"💎 cam_token mean: {model.visual_geometry_transformer.cam_token.mean().item():.8f}")
    print(f"💎 reg_token mean: {model.visual_geometry_transformer.reg_token.mean().item():.8f}")
    
    # Check a few weights to see if they are not just zeros/trash
    print(f"💎 cam_head layer1 weight norm: {model.cam_head.refine_net[0].norm1.weight.norm().item():.4f}")
    
    # Run a dummy forward
    print("🚀 Running dummy forward...")
    dummy_input = {
        'img': torch.randn(1, 1, 3, 518, 518)
    }
    with torch.no_grad():
        try:
            out = model(dummy_input)
            print("✅ Forward pass successful")
            print(f"   - Predicted camera: {out.get('camera_params')}")
            if 'pts3d' in out:
                print(f"   - PTS3D range: [{out['pts3d'].min().item():.2f}, {out['pts3d'].max().item():.2f}]")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")

if __name__ == "__main__":
    diagnostic()
