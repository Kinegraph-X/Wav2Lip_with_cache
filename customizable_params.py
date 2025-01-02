from args_parser import args_parser

class HParams:
	def __init__(self, **kwargs):
		self.data = {}

		for key, value in kwargs.items():
			self.data[key] = value

	def __getattr__(self, key):
		if key not in self.data:
			raise AttributeError("'HParams' object has no attribute %s" % key)
		return self.data[key]

	def set_hparam(self, key, value):
		self.data[key] = value

if (args_parser.avatar_type == 'generic_man'):
	video_file_path = 'Avatar_generic_man.mp4'
elif (args_parser.avatar_type == 'generic_woman'):
	video_file_path = 'Avatar_generic_woman.mp4'
elif (args_parser.avatar_type == 'EBU_n19'):
	video_file_path = 'Avatar_Small_Online.mp4'

# Default hyperparameters
customizable_params = HParams(
	static_video_file_path = video_file_path
)