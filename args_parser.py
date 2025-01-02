import argparse

parser = argparse.ArgumentParser(description='Worker which records an audio file from the mic, sneds it to a Google Notebook and retrieve the result of the computation')
parser.add_argument('--avatar_type', type=str, 
					help='avatar_type defines the video file serving as base for lip-sync',
                    required=False
                    )
args = parser.parse_args()