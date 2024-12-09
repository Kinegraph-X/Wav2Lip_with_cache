import os, cgi


class Args_parser():
	media_folder = "media/"
	default_params = {
		"fps" : 25,
		"resize_factor" : 1,
		"rotate" : 0,
		"crop" : [0, -1, 0, -1],
		"img_size" : 96,
		"face_det_batch_size" : 16,
		"pads" : [0, 10, 0, 0],
		"nosmooth" : False,
		"checkpoint_path" : "checkpoint_path/wav2lip_gan.pth"

	}
	# def __init__(self, req):

	def parse(self, req):
		content_type, params_dict = cgi.parse_header(req.headers['content-type'])
		
		if content_type == 'multipart/form-data':
			# Parse the form-data
			form = cgi.FieldStorage(
				fp=req.rfile,
				headers=req.headers,
				environ={'REQUEST_METHOD': 'POST',
						 'CONTENT_TYPE': req.headers['Content-Type']})
			
			# Extract parameters
			self.params = {}
			for key in form.keys():
				field_item = form[key]
				if field_item.filename:  # It's a file
					filename = os.path.basename(field_item.filename)
					self.params["audio_filename"] = filename
					file_data = field_item.file.read()
					
					# Save the file
					with open(f"{self.media_folder}{filename}", 'wb') as f:
						f.write(file_data)
					print(f"File '{filename}' received and saved.")
				else:
					self.params[key] = form.getvalue(key)

		# merge with default params
		self.params = self.default_params | self.params
			
			

args_parser = Args_parser()