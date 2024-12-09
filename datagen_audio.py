import cv2, numpy as np
from models.wav2lip_cache import Wav2LipCache
from hparams import hparams

# mock args for now...
class Args:
	# fps = 25
	# resize_factor = 1
	# rotate = 0
	# crop = [0, -1, 0, -1]
	# face_det_batch_size = 128
	# pads = [0, 10, 0, 0]
	# nosmooth = False
	# box = [-1, -1, -1, -1]
	img_size = 96
	# batch_size = 128
	
args = Args()






def start(mel_chunks):
    batch_size = hparams.video_batch_size
    """
    Generates batches of mel-spectrogram chunks.
    
    Args:
        mel_chunks (list): List of mel-spectrogram chunks.
        batch_size (int): Number of mel chunks per batch.
    
    Yields:
        List of reshaped mel-spectrogram batches.
    """
    mel_batch = []
    for mel in mel_chunks:
        mel_batch.append(mel)

        if len(mel_batch) >= batch_size:
            mel_batch = np.asarray(mel_batch)

            # Reshape mel batch for model input
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield mel_batch
            mel_batch = []

    if mel_batch:  # Handle the last batch if it's not full
        mel_batch = np.asarray(mel_batch)
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield mel_batch