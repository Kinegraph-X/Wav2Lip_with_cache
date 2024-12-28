import torch, face_detection, cv2, time
import numpy as np
from tqdm import tqdm
from models.wav2lip_cache import Wav2LipCache
from hparams import hparams
from logger import logger
from args_parser import args_parser

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info('Using {} for inference.'.format(device))

"""
# mock args for now...
class Args:
	# fps = 25
	# resize_factor = 1
	# rotate = 0
	# crop = [0, -1, 0, -1]
	img_size = 96
	face_det_batch_size = 16
	pads = [0, 10, 0, 0]
	nosmooth = False
	
args = Args()
"""

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def start(images):
	
	cache = Wav2LipCache("cache/face_detection")
	
	if cache.is_cached(args_parser.params['video_file_path'], "face_detection"):
		logger.info("Face_detection file is cached")
		results = cache.read_npy(args_parser.params['video_file_path'], "face_detection")
		logger.debug(f"loaded images length is {len(images)}")

	else:
		logger.debug(f"computed images length is {len(images)}. Starting face detection...")

		detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
												flip_input=False, device=device)

		batch_size = args_parser.params["face_det_batch_size"]
		results = []

		while 1:
			predictions = []
			try:
				for i in tqdm(range(0, len(images), batch_size)):
					predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
			except RuntimeError:
				if batch_size == 1: 
					raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
				# batch_size //= 2
				# print('Recovering from OOM error; New batch size: {}'.format(batch_size))
				logger.error('Recovering from OOM error; It\'s frequently related to available memory being too low. Aborting...')
				return results
			break

		pady1, pady2, padx1, padx2 = args_parser.params["pads"]
		for rect, image in zip(predictions, images):
			if rect is None:
				cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
				raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

			y1 = max(0, rect[1] - pady1)
			y2 = min(image.shape[0], rect[3] + pady2)
			x1 = max(0, rect[0] - padx1)
			x2 = min(image.shape[1], rect[2] + padx2)
			
			results.append([x1, y1, x2, y2])

		boxes = np.array(results)
		
		start_time = time.perf_counter()
		if not args_parser.params["nosmooth"]: boxes = get_smoothened_boxes(boxes, T=5)
		end_time = time.perf_counter()
		logger.debug(f'Smoothing took {end_time - start_time}')
		
		# results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
		
		idx = 0
		for image, box in zip(images, boxes):  # Loop over images and corresponding boxes
			x1, y1, x2, y2 = box  # Unpack the box coordinates
			# hack to avoid the beard not being detected (it doesn't really work)
			y2 = min(y2 + 5, image.shape[0])
			cropped_image = image[y1:y2, x1:x2]  # Crop the image using slicing
			resized_image = cv2.resize(cropped_image, (args_parser.params["img_size"], args_parser.params["img_size"]))  # Resize to target size
			result = [resized_image, (y1, y2, x1, x2)]  # Create a list with the resized image and box
			results[idx] = result
			idx += 1

		resized_images = np.array([result[0] for result in results])  # Shape: (N, H, W, C)
		bounding_boxes = np.array([result[1] for result in results])  # Shape: (N, 4)

		results = np.array([(img, box) for img, box in zip(resized_images, bounding_boxes)], dtype=object)

		del detector

		logger.info("Face_detection file is not cached")
		cache.write_npy(args_parser.params['video_file_path'], 'face_detection', results)

	return results 