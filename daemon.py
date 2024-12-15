from flask import Flask, request, send_file, Response
import wave
import os, time
import numpy as np
from process_Wav2Lip import process, new_batch_available, status, processing_ended
from args_parser import args_parser
from hparams import hparams
from serializer import serialize_chunk
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Flask app
app = Flask(__name__)

# Global variables
wf = None
streamed = False
media_folder = "./media/"  # folder for saving files
sent_frames = 0

def handle_chunked_audio(request, audio_chunk):
	global wf
	# Initialize or write to the WAV file
	if wf is None:
		file_path = os.path.join(media_folder, request.headers.get("X-Audio-Filename"))
		"""
		if os.path.exists(file_path):
			print(f'Wavefile removed')
			os.remove(file_path)
		"""
		print(f'Wavefile created : {request.headers.get("X-Audio-Filename")}')
		wf = wave.open(file_path, 'wb')
		wf.setnchannels(int(request.headers.get("X-Channels")))
		wf.setsampwidth(2)  # 16-bit PCM
		wf.setframerate(int(request.headers.get("X-Sample-Rate")))

	if not request.headers.get("X-Audio-Chunk-Timestamp") == 'EOF':
		if len(audio_chunk):
			wf.writeframes(audio_chunk)
	else:
		wf.close()
		wf = None

@app.route('/', methods=['GET'])
def handle_get():
	
	args_parser.parse(request)
	if request.args.get('path'):
		file_path = args_parser.params["path"]
		if os.path.exists(file_path):
			return send_file(file_path, mimetype="application/octet-stream")
		return "File not found.", 404

	if request.args.get("next_batch"):
		# print('polling request received')
		return long_polling()

	return "Request received", 200

@app.route('/', methods=['POST'])
def handle_post():
	""" POST requests are for chunked audio data."""
	global wf, streamed, sent_frames
	# """
	args_parser.parse(request)

	timestamp = request.headers.get("X-Audio-Chunk-Timestamp")
	content_length = request.content_length or 0

	if not request.headers.get("X-Audio-Filename"):
		return "Audio file not received, aborting...", 200

	args_parser.params["audio_filename"] = request.headers.get("X-Audio-Filename")
	if timestamp and timestamp != 'EOF':
		streamed = True
		if content_length > 0:
			handle_chunked_audio(request, request.data)
			return f"Received chunk: {timestamp}", 200
	elif timestamp == 'EOF':
		wf = None
		sent_frames = 0
		status["current_frame_count"] = 0
		process(streamed)
		return f'Completed processing new audio file: {request.headers.get("X-Audio-Filename")}', 200

	# """
	return "Invalid Request", 400

def long_polling():
	global sent_frames, current_frame_count
	""" long polling for video chunks."""
	timeout = 30
	start_time = time.time()

	if processing_ended.is_set():
		processing_ended.clear()
		return 'processing_ended', 200, {"Content-Type": "text/plain"}

	while not new_batch_available.is_set():
		if time.time() - start_time < timeout:
			time.sleep(.1)
		else:
			return 'long_polling_timeout', 200, {"Content-Type": "text/plain"}

	
	processed_frames = np.load(hparams.temp_pred_file_path)
	# print(f'{status["current_frame_count"]} {len(processed_frames)}')
	if status["current_frame_count"] == len(processed_frames):
		processing_ended.set()

	# print(processed_frames.shape)
	current_cursor = sent_frames
	print(f'new batch yielded, sneding response for frame idx : {current_cursor}')
	sent_frames = len(processed_frames)
	new_batch_available.clear()

	data = processed_frames[current_cursor:]
	serialized_chunk = serialize_chunk(data.shape, current_cursor, data)
	return Response(
		serialized_chunk,
		content_type='application/octet-stream',
		headers={'Content-Disposition': 'attachment; filename="chunk.bin"'}
	)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=3000, threaded=True)
