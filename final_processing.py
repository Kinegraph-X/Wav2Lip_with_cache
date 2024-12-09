import subprocess, platform, cv2, torch, numpy as np
from tqdm import tqdm
import datagen_audio
from hparams import hparams
from models.wav2lip_cache import Wav2LipCache
from models.wav2lip_compute_embeddings import Wav2Lip

from args_parser import args_parser

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
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def start(full_frames, mel_chunks, face_detect_results):
	batch_size = hparams.video_batch_size

	# for i, (mel_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
	total_batches = int(np.ceil(float(len(mel_chunks)) / batch_size))
	gen = datagen_audio.start(mel_chunks)

	for i, mel_batch in zip(range(total_batches), gen):
		
		if i == 0:
			# model = load_model(args_parser.params["checkpoint_path"])
			model = load_model(hparams.checkpoint_path, hparams.static_video_file_path)
			print ("Model loaded")
			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter('temp/result.avi', 
										cv2.VideoWriter_fourcc(*'DIVX'), args_parser.params["fps"], (frame_w, frame_h))
			
		mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

		with torch.no_grad():
			pred = model(mel_batch, i, batch_size)

		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

		for p in pred:
			y1, y2, x1, x2 = face_detect_results[i][1]
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
			f = full_frames[i]
			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()
	
	output_path = "output/test_241206.mp4"
	command = '{}ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(hparams.ffmpeg_path, hparams.media_folder + args_parser.params["audio_filename"], 'temp/result.avi', output_path)
	subprocess.call(command, shell=platform.system() != 'Windows', stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
	# subprocess.call(command, shell=platform.system() != 'Windows', stderr=subprocess.STDOUT)
	print(f'Video file saved to {output_path}')