import os, cgi
from urllib.parse import urlparse, parse_qs
from customizable_params import customizable_params
from logger import logger

class Args_parser():
	media_folder = "media/"
	default_params = {
		"fps" : 25,
		"resize_factor" : 1,
		"rotate" : 0,
		"crop" : [0, -1, 0, -1],
		"img_size" : 96,
		"face_det_batch_size" : 16,
		"pads" : [0, 0, 0, 0],
		"nosmooth" : False,
		"warm_start" : False,
		"checkpoint_path" : "checkpoint_path/wav2lip_gan.pth",
		"video_file_path" : customizable_params.static_video_file_path,
		"generic_man_video_file_path" : 'Avatar_generic_man.mp4',
		"generic_woman_video_file_path" : 'Avatar_generic_woman.mp4',
		"EBU_n19_video_file_path" : 'Avatar_Small_Online.mp4',
		"audio_filename" : "test_medium.wav"
	}
	def __init__(self):
		self.params = self.default_params
		self.headers = {}

	def parse(self, req):
		
		if req.headers.get('Content-Type') is not None:
			# content_type, params_dict = cgi.parse_header(req.headers['Content-Type'])
			content_type = req.headers.get('Content-Type')
			
			if content_type == 'multipart/form-data':
				# Parse the form-data
				form = cgi.FieldStorage(
					fp=req.rfile,
					headers=req.headers,
					environ={'REQUEST_METHOD': 'POST',
							'CONTENT_TYPE': req.headers['Content-Type']})
				
				# Extract parameters
				for key in form.keys():
					field_item = form[key]
					if field_item.filename:  # It's a file
						filename = os.path.basename(field_item.filename)
						self.params["audio_filename"] = filename
						file_data = field_item.file.read()
						
						# Save the file
						with open(f"{self.media_folder}{filename}", 'wb') as f:
							f.write(file_data)
						logger.info(f"File '{filename}' received and saved.")
					else:
						if form.getvalue(key) == 'True':
							self.params[key] = True
						elif form.getvalue(key) == 'False':
							self.params[key] = False
						else:
							self.params[key] = form.getvalue(key)

			elif content_type == 'application/octet-stream':
				for key, value in req.headers.items():
					self.headers[key] = value

		# merge with default params
		self.params = self.default_params | self.params
			
			

args_parser = Args_parser()