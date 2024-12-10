import custom_libs.audio as audio
from hparams import hparams
import time

from args_parser import args_parser

"""
# mock args for now...
class Args:
	fps = 25
	
args = Args()
"""


mel_step_size = 16

def start(full_frames):
	init_time = time.perf_counter()

	wav = audio.load_wav(hparams.media_folder + args_parser.params["audio_filename"], 16000)
	# wav = audio.load_audio(args_parser.params["audio_filename"], 16000)
	# wav = audio.torch_load_audio(args_parser["audio_filename"], 16000)
	end_time = time.perf_counter()
	print(f'Audio loading took {end_time - init_time}')

	init_time = time.perf_counter()
	mel = audio.melspectrogram(wav)
	end_time = time.perf_counter()
	print(f'mel shape is {mel.shape}')
	print(f'mel spectrogram took {end_time - init_time}')

	mel_chunks = []
	mel_idx_multiplier = 80./args_parser.params["fps"] 
	i = 0
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1

	print("Length of mel chunks: {}".format(len(mel_chunks)))

	full_frames = full_frames[:len(mel_chunks)]

	return full_frames, mel_chunks