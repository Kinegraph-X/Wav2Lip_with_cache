import cv2, os, time
from hparams import hparams
from models.wav2lip_cache import Wav2LipCache
from logger import logger
from args_parser import args_parser

"""
# mock args for now...
class Args:
	resize_factor = 1
	rotate = 0
	crop = [0, -1, 0, -1]
	
args = Args()
"""

def start():
	cache = Wav2LipCache("cache/raw_frames")
	
	# if not os.path.isfile(agrs_parser.params.video_file_path):
	if not os.path.isfile(args_parser.params["video_file_path"]):
		raise ValueError('--face argument must be a valid path to video/image file')

	# elif agrs_parser.params.video_file_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
	elif args_parser.params['video_file_path'].split('.')[1] in ['jpg', 'png', 'jpeg']:
		full_frames = [cv2.imread(args_parser.params['video_file_path'])]
		fps = args_parser.params["fps"]

	elif cache.is_cached(args_parser.params['video_file_path'], "raw_frames"):
		logger.info("Frames file is cached")
		full_frames = cache.read_npy(args_parser.params['video_file_path'], "raw_frames")
	
	else:
		# video_stream = cv2.VideoCapture(args_parser.params['video_file_path'])
		start_time = time.perf_counter()
		video_stream = cv2.VideoCapture(args_parser.params['video_file_path'])
		end_time = time.perf_counter()
		logger.debug(f'VideoCapure init took : {end_time - start_time}')
		args_parser.params["fps"] = fps = video_stream.get(cv2.CAP_PROP_FPS)

		logger.info('Reading video frames...')

		start_time = time.perf_counter()
		first_frame_seen = False
		full_frames = []
		while 1:
			still_reading, frame = video_stream.read()

			if not still_reading:
				video_stream.release()
				break
			if args_parser.params["resize_factor"] != 1:
				frame = cv2.resize(frame, (frame.shape[1]//args_parser.params["resize_factor"], frame.shape[0]//args_parser.params["resize_factor"]))

			if args_parser.params["rotate"]:
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

			y1, y2, x1, x2 = args_parser.params["crop"]
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]

			if not first_frame_seen:
				first_frame_seen = True

			frame = frame[y1:y2, x1:x2]

			full_frames.append(frame)

		end_time = time.perf_counter()
		logger.debug(f'Effective VideoCapure took : {end_time - start_time}')

		logger.info("Frames file is not cached")
		cache.write_npy(args_parser.params['video_file_path'], 'raw_frames', full_frames)

	logger.debug("Number of frames available for inference: "+str(len(full_frames)))
	return full_frames