import os, threading
import numpy as np
from hparams import hparams
import prepare_video
import face_detect
import build_mels
import image_embeddings_preprocess
import final_processing
import time

warm_start = {

}

status = {
	"current_frame_count" : 0,
	"processed_frames" : np.empty((0, 270, 480, 3), dtype = np.uint8)
	}
new_batch_available = threading.Event()
processing_ended = threading.Event()

def save_pred_incrementally(pred):
		"""
		if os.path.exists(hparams.temp_pred_file_path):
				# Load existing file and append new data
				existing_data = np.load(hparams.temp_pred_file_path)
				new_data = np.concatenate((existing_data, pred), axis=0)
		else:
				# Start a new file
				new_data = pred
		"""

		status["processed_frames"] = np.concatenate((status["processed_frames"], pred), axis = 0)
		# Save back the combined data
		# np.save(hparams.temp_pred_file_path, new_data)
		new_batch_available.set()

def process_warmed_up(streamed = False):
		global current_frame_count
		# """
		start_time = time.perf_counter()

		# mel spectrogram may fail is the file is well formatted but too short
		try:
			_, mel_chunks = build_mels.start(warm_start["frames"])
		except Exception as e:
			return e.args
		
		status["current_frame_count"] = len(mel_chunks)
		preds = final_processing.start(warm_start["frames"], mel_chunks, warm_start["face_detect_results"], streamed)

		if streamed:
			for pred in preds:
				save_pred_incrementally(pred)

		end_time = time.perf_counter()
		print(f'Total script took {end_time - start_time}')
		return "Processing succeeded"
		# """

def process_cold_start(streamed = False):
		global status
		# """
		start_time = time.perf_counter()
		frames = prepare_video.start()
		warm_start["frames"] = frames
		face_detect_results = face_detect.start(frames)
		warm_start["face_detect_results"] = face_detect_results
		image_embeddings_preprocess.start(frames)

		# mel spectrogram may fail is the file is well formatted but too short
		try:
			_, mel_chunks = build_mels.start(warm_start["frames"])
		except Exception as e:
			return e.args
		
		status["current_frame_count"] = len(mel_chunks)
		preds = final_processing.start(frames, mel_chunks, face_detect_results, streamed)

		if streamed:
			for pred in preds:
					# print(f'shape of pred yielded {len(pred)}')
					save_pred_incrementally(pred)

		end_time = time.perf_counter()
		print(f'Total script took {end_time - start_time}')
		return "Processing succeeded"
		# """

def process(streamed = False):
		if os.path.exists(hparams.temp_pred_file_path):
			os.remove(hparams.temp_pred_file_path)

		if (len(warm_start) > 0):
				return process_warmed_up(streamed)
		else:
				return process_cold_start(streamed)