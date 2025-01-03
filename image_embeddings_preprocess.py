import torch, numpy as np
from tqdm import tqdm
import datagen_images
from hparams import hparams
from models.wav2lip_cache import Wav2LipCache
from models.wav2lip_image_embeddings import Wav2Lip
from logger import logger
from http_args_parser import args_parser

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
	logger.info("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def start(full_frames, avatar_type):
	batch_size = hparams.video_batch_size
	video_file_path = hparams.media_folder + args_parser.params[avatar_type + "_video_file_path"]

	cache = Wav2LipCache('cache/embeddings')
	if cache.is_cached(video_file_path, "embeddings"):
		logger.info("Will use precomputed embeddings...")
		return
	else:
		logger.info("No precomputed embeddings found. Proceeding without cache.")

		gen = datagen_images.start(full_frames.copy(), avatar_type)

		for i, (img_batch) in enumerate(tqdm(gen, total=len(full_frames))):
			# print(f'index is {i}')
			if i == 0:
				# model = load_model(args_parser.params["checkpoint_path"])
				model = load_model(args_parser.params["checkpoint_path"], video_file_path)
				logger.info ("Model loaded")

			img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)

			with torch.no_grad():
				model(img_batch, i, len(full_frames) - 1)