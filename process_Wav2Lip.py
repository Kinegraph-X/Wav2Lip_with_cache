import os, threading
import numpy as np
from hparams import hparams
import prepare_video
import face_detect
import build_mels
import image_embeddings_preprocess
import final_processing
import time
from models.wav2lip_cache import Wav2LipCache
from logger import logger
from http_args_parser import args_parser

warm_start = {

}

status = {
	"current_frame_count" : 0,
	"processed_frames" : None
	}
new_batch_available = threading.Event()
processing_ended = threading.Event()

def save_pred_incrementally(pred, avatar_type = ''):

		video_path = hparams.media_folder + args_parser.params[avatar_type + '_video_file_path']
		cache = Wav2LipCache('cache/raw_frames')
		cached_data =  cache.read_npy(video_path, 'raw_frames')

		if status["processed_frames"] is None:
			status["processed_frames"] = np.empty((0, cached_data.shape[1], cached_data.shape[2], 3), dtype = np.uint8)

		# print(status["processed_frames"].shape)
		# print(pred.shape)

		status["processed_frames"] = np.concatenate((status["processed_frames"], pred), axis = 0)
		# Save back the combined data
		# np.save(hparams.temp_pred_file_path, new_data)
		new_batch_available.set()

def process_warmed_up(streamed = False, avatar_type = ''):
		global current_frame_count
		# """
		start_time = time.perf_counter()

		if not avatar_type:
			return "Processing cancelled"

		# mel spectrogram may fail is the file is well formatted but too short
		try:
			_, mel_chunks = build_mels.start(warm_start["frames"])
		except Exception as e:
			return e.args
		
		status["current_frame_count"] = len(mel_chunks)
		preds = final_processing.start(warm_start["frames"], mel_chunks, warm_start["face_detect_results"], streamed, avatar_type)

		if streamed:
			for pred in preds:
				save_pred_incrementally(pred, avatar_type)

		end_time = time.perf_counter()
		logger.debug(f'Total script took {end_time - start_time}')
		return "Processing succeeded"
		# """

def process_cold_start(streamed = False, avatar_type = ''):
		global status
		warm_start["avatar_type"] = avatar_type

		start_time = time.perf_counter()

		if not avatar_type:
			return "Processing cancelled"
		
		frames = prepare_video.start(avatar_type)
		warm_start["frames"] = frames
		face_detect_results = face_detect.start(frames, avatar_type)
		if not len(face_detect_results):
			return "processing aborted due to an error in face detection : empty data returned"
		warm_start["face_detect_results"] = face_detect_results
		image_embeddings_preprocess.start(frames, avatar_type)

		

		# mel spectrogram may fail is the file is well formatted but too short
		try:
			_, mel_chunks = build_mels.start(warm_start["frames"])
		except Exception as e:
			return e.args
		
		status["current_frame_count"] = len(mel_chunks)
		preds = final_processing.start(frames, mel_chunks, face_detect_results, streamed, avatar_type)

		if streamed:
			for pred in preds:
					# print(f'shape of pred yielded {len(pred)}')
					save_pred_incrementally(pred, avatar_type)

		end_time = time.perf_counter()
		logger.debug(f'Total script took {end_time - start_time}')

		return "Processing succeeded"

def process(streamed = False, avatar_type = ''):
		if os.path.exists(hparams.temp_pred_file_path):
			os.remove(hparams.temp_pred_file_path)

		if (len(warm_start) > 1 and warm_start["avatar_type"] == avatar_type):
				return process_warmed_up(streamed, avatar_type)
		else:
				return process_cold_start(streamed, avatar_type)