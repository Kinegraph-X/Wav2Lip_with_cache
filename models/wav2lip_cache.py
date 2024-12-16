import os
import torch
import numpy as np
import hashlib

class Wav2LipCache:
    def __init__(self, cache_dir):
        if not cache_dir:
            raise Exception("Wav2Lip_cache must be passsed a path to a cache folder")

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def switch_folder(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, video_path, type, idx = "master"):
        if not type=="embeddings":
            ext = 'npy'
        else:
            ext = "pt"
        # Generate a unique identifier for the video file based on its content
        video_hash = hashlib.md5(open(video_path, 'rb').read()).hexdigest()
        return os.path.join(self.cache_dir, f"{video_hash}_{type}_{idx}.{ext}")

    def save_embeddings(self, embeddings, video_path, idx = "master"):
        """Save face embeddings to the cache."""
        cache_path = self._get_cache_path(video_path, "embeddings", idx)
        torch.save(embeddings, cache_path)
        print(f"Embeddings saved to {cache_path}")

    def load_embeddings(self, video_path, idx = "master"):
        """Load face embeddings from the cache."""
        cache_path = self._get_cache_path(video_path, "embeddings", idx)
        if os.path.exists(cache_path):
            # print(f"Loading embeddings from {cache_path}")
            if torch.cuda.is_available():
                return torch.load(cache_path, map_location=torch.device('cuda:0'))
            else:
                return torch.load(cache_path)
        return None

    def is_cached(self, video_path, type, idx = "master"):
        """Check if embeddings are cached."""
        cache_path = self._get_cache_path(video_path, type, idx)
        return os.path.exists(cache_path)
    
    def read_npy(self, video_path, type, idx = "master"):
        cache_path = self._get_cache_path(video_path, type, idx)
        if os.path.exists(cache_path):
            # fp = open(cache_path, "r")
            result = np.load(cache_path, allow_pickle=True)
            # fp.close()
            return result
        else:
            raise Exception(f"cache folder not found {self.cache_dir}")
        
    def write_npy(self, video_path, type, data, idx = "master"):
        cache_path = self._get_cache_path(video_path, type, idx)
        # fp = open(cache_path, "w")
        np.save(cache_path, data)
        # fp.close()
        print(f"{type} saved to {cache_path}")

# Example usage
def cache_face_encoder_output(model, video_frames, video_path):
    """
    Cache the output of face_encoder_blocks for a given video.
    - model: Wav2Lip model.
    - video_frames: Preprocessed frames as tensors (B, C, H, W).
    - video_path: Path to the video file for caching.
    """
    cache = Wav2LipCache()

    # Check if embeddings are already cached
    if cache.is_cached(video_path):
        return cache.load_embeddings(video_path)

    # Process the frames through face_encoder_blocks
    embeddings = []
    x = video_frames
    for block in model.face_encoder_blocks:
        x = block(x)
        embeddings.append(x)

    # Combine embeddings into a single tensor for saving
    embeddings = torch.stack(embeddings, dim=0)  # Shape: (num_blocks, B, C, H, W)
    cache.save_embeddings(embeddings, video_path)

    return embeddings
