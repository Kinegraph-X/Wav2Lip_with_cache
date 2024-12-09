import cv2
import numpy as np
from hparams import hparams
from models.wav2lip_cache import Wav2LipCache

# mock args for now...
class Args:
	# fps = 25
	# resize_factor = 1
	# rotate = 0
	# crop = [0, -1, 0, -1]
	# face_det_batch_size = 128
	# pads = [0, 10, 0, 0]
	# nosmooth = False
	box = [-1, -1, -1, -1]
	img_size = 96
	
args = Args()

def start(frames, mels=None):
	cache = Wav2LipCache("cache/face_detection")
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	"""
	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect([frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
	"""

	if cache.is_cached(hparams.static_video_file_path, "face_detection"):
		print("Face_detection file is cached")
		face_det_results = cache.read_npy(hparams.static_video_file_path, "face_detection")

	else:
		print("Face_detection file should be cached")
	
	print(f'Length of frames array {len(frames)}')

	for i, m in enumerate(mels):
		idx = 0 if args.static else i % len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch
