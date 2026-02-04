import torch
import numpy as np

def distribute_data_to_gpus(data, dim, rank, local_rank, num_gpus, dtype):
    """
    Mock version for single GPU inference. Returns the data as is.
    """
    # Simply move data to the correct device/dtype
    local_data = data.to(device=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu", dtype=dtype)
    return local_data


def gather_results(local_result, dim, world_size):
    """
    Mock version for single GPU inference. Returns local_result as gathered result.
    """
    return local_result

def export_to_video(video_frames, output_video_path, fps=12):
    # This export function might not be used in ComfyUI optimization flow but keeping it just in case
    import imageio
    assert all(
        isinstance(frame, np.ndarray) for frame in video_frames
    ), "All video frames must be NumPy arrays."
    if not output_video_path.endswith(".mp4"):
        output_video_path += ".mp4"
    with imageio.get_writer(output_video_path, fps=fps, format="mp4") as writer:
        for frame in video_frames:
            writer.append_data((frame * 255).astype(np.uint8))
