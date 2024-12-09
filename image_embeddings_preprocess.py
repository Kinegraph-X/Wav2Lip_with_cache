import torch, numpy as np
from tqdm import tqdm
import datagen_images
from hparams import hparams
from models.wav2lip_cache import Wav2LipCache
from models.wav2lip_image_embeddings import Wav2Lip

from args_parser import args_parser

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
# mock args for now...
class Args:
	fps = 25
	img_size = 96
	
args = Args()
"""

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path, video_path):
	model = Wav2Lip(video_path)
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def start(full_frames, mel_chunks):
	batch_size = hparams.video_batch_size

	cache = Wav2LipCache('cache/embeddings')
	if cache.is_cached(hparams.static_video_file_path, "embeddings"):
		print("Will use precomputed embeddings...")
		return
	else:
		print("No precomputed embeddings found. Proceeding without cache.")

		gen = datagen_images.start(full_frames.copy())

		for i, (img_batch) in enumerate(tqdm(gen, total=len(full_frames))):
			print(f'index is {i}')
			if i == 0:
				# model = load_model(args_parser.params["checkpoint_path"])
				model = load_model(args_parser.params["checkpoint_path"], hparams.static_video_file_path)
				print ("Model loaded")

			img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)

			with torch.no_grad():
				model(img_batch, i, len(full_frames) - 1)