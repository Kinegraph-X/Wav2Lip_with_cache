import asyncio
import nest_asyncio
from flask import Flask
from pyngrok import ngrok  # Use pyngrok for a simpler API
import os
from dotenv import load_dotenv

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

# Set Ngrok auth token
load_dotenv()

# Flask app
app = Flask(__name__)

# Global variables
wf = None
streamed = False
media_folder = "./media/"  # folder for saving files
sent_frames = 0

# Enable nested event loops (required for Colab)
nest_asyncio.apply()

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
	global start_time

	args_parser.parse(request)
	if request.args.get('path'):
		file_path = args_parser.params["path"]
		if os.path.exists(file_path):
			return send_file(file_path, mimetype="application/octet-stream")
		return "File not found.", 404

	if request.args.get("next_batch"):
		start_time = time.perf_counter()
		print('polling request received')
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
		status["processed_frames"] = None
		process(streamed)
		return f'Completed processing new audio file: {request.headers.get("X-Audio-Filename")}', 200

	# """
	return "Invalid Request", 400

def long_polling():
	global sent_frames, current_frame_count, start_time
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

	# print(f'new batch yielded. state of processing_ended : {processing_ended.is_set()}')
	# processed_frames = np.load(hparams.temp_pred_file_path)
	processed_frames = status["processed_frames"]
	
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

async def main():
    upload_port = 5000

    # Start the Flask app on a specific port
    flask_task = asyncio.create_task(asyncio.to_thread(app.run, host="0.0.0.0", port=upload_port, debug=False, use_reloader=False))

    # Start Ngrok tunnel
    print("Starting Ngrok tunnel...")
    ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if ngrok_auth_token:
        ngrok.set_auth_token(ngrok_auth_token)

    public_url = ngrok.connect(upload_port).public_url
    print(f"Ngrok tunnel started! Public URL: {public_url}")

    # Wait for server to run until interrupted
    try:
        await flask_task
    except asyncio.CancelledError:
        print("\nServer interrupted by user.")
    finally:
        print("Shutting down Flask server and Ngrok tunnel...")
        ngrok.disconnect(public_url)
        ngrok.kill()
		
# Run the main function in Jupyter
if __name__ == "__main__":
# Use the existing event loop
    loop = asyncio.get_event_loop()
    upload_task = loop.create_task(main())  # Start the task in the background

    try:
        loop.run_until_complete(upload_task)  # Run the event loop until the task completes
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, shutting down gracefully.")
        # tasks = asyncio.all_tasks(loop)
        """
        for task in tasks:
            task.cancel()
        """

        upload_task.cancel()  # Attempt to cancel the task gracefully
        # loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        # loop.close()
        print("Cleanup complete.")
    except Exception as e:
        print(f"An error occurred: {e}")
