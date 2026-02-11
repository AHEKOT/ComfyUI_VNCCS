
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

nodes_to_test = [
    ".character_creator",
    ".character_selector",
    ".dataset_generator",
    ".emotion_generator",
    ".emotion_generator_v2",
    ".sheet_crop",
    ".sheet_manager",
    ".sprite_generator",
    ".vnccs_pipe",
    ".vnccs_qwen_encoder",
    ".sampler_scheduler_picker",
    ".common_nodes",
    ".pose_generator",
    ".vnccs_utils",
    ".background_generator",
    ".vnccs_panorama_mapper",
    ".character_creator_v2",
    ".character_cloner",
    ".clothes_designer",
    ".sprite_manager"
]

print("🔍 Testing VNCCS Node Imports...")

for node in nodes_to_test:
    try:
        # We need to simulate the package import
        # Instead of relative import, we use absolute if possible or mock the package
        module_path = f"nodes{node}"
        print(f"  Testing {module_path}...", end=" ", flush=True)
        __import__(module_path)
        print("✅ OK")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()

print("\nDone.")
