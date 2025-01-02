import cv2, numpy as np
from models.wav2lip_cache import Wav2LipCache
from hparams import hparams
from logger import logger
from http_args_parser import args_parser

"""
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
"""

def start(frames, avatar_type = ''):
	# batch_size = hparams.video_batch_size

	"""
	Generates batches of resized images for face regions.
	
	Args:
		frames (list): List of video frames (images).
		batch_size (int): Number of images per batch.
		static_frame (bool): If True, use only the first frame repeatedly.
	
	Yields:
		List of resized and normalized face images for the batch.
	"""
	cache = Wav2LipCache("cache/face_detection")
	video_file_path = hparams.media_folder + args_parser.params[avatar_type + '_video_file_path']

	if cache.is_cached(video_file_path, "face_detection"):
		logger.info("Face_detection file is cached")
		face_det_results = cache.read_npy(video_file_path, "face_detection")

	else:
		raise Exception("Face_detection file should be cached")

	img_batch = []
	for i, (face, _) in enumerate(face_det_results):
		face = cv2.resize(face, (args_parser.params["img_size"], args_parser.params["img_size"]))
		img_batch.append(face)
		
		# if len(img_batch) >= batch_size:
		img_batch = np.asarray(img_batch)

		# Normalize and prepare batch
		img_masked = img_batch.copy()
		img_masked[:, args_parser.params["img_size"] // 2:] = 0  # Mask the lower half of the images
		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0

		yield img_batch
		img_batch = []
		

	if img_batch:  # Handle the last batch if it's not full
		img_batch = np.asarray(img_batch)
		img_masked = img_batch.copy()
		img_masked[:, args_parser.params["img_size"] // 2:] = 0
		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
		yield img_batch