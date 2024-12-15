# import time
import wave
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from args_parser import args_parser

from process_Wav2Lip import process


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

streamed = False
wf = None

def long_polling(preds):
    return


def handle_chuncked_wavefile(data):
    global wf
    if wf is None:
        print(f'Wavefile created : {args_parser.headers["audio_filename"]}')
        wf = wave.open(f'{args_parser.media_folder}{args_parser.headers["audio_filename"]}', 'w')
        wf.setnchannels(int(args_parser.headers["channels"]))
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(int(args_parser.headers["sample_rate"]))
    
    # print(len(data))
    if not args_parser.headers['audio_chunk_timestamp'] == 'EOF':
        if len(data):
            wf.writeframes(data)
    else:
        wf.close()
        wf = None


# Custom HTTP handler
class HelloHandler(BaseHTTPRequestHandler):
    
    # protocol_version = 'HTTP/1.1'

    def do_GET(self):
        args_parser.parse(self)
        
        self.send_response(200)
        
        if "path" in args_parser.params:
            self.send_header("Content-type", "application/octet-stream")
            self.end_headers()
            with open(args_parser.params["path"], 'rb') as file: 
                self.wfile.write(file.read())
        elif args_parser.params['chunk_id']:
            long_polling(args_parser.params['chunk_id'])
        else:
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            # message = process()
            message = "request received"
            self.wfile.write(f'{message}'.encode())

    def do_POST(self):
        print("received request")
        # start_time = time.perf_counter()
        
        args_parser.parse(self)

        if not args_parser.params["audio_filename"] and not args_parser.headers['audio_filename']:
            print("Audio file not received, aborting...")
            self.send_response(500)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Audio file not received, aborting...")
        
        if args_parser.headers['audio_filename'] and args_parser.headers['audio_chunk_timestamp'] is not None:
            streamed = True
            content_length = int(args_parser.headers.get('Content-Length'))
            handle_chuncked_wavefile(self.rfile.read(content_length))

            if not args_parser.headers['audio_chunk_timestamp'] == 'EOF':
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(f'received chunk : {args_parser.headers["audio_chunk_timestamp"]}'.encode())
            
            else:
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(f'Completed saving new audio file : {args_parser.headers["audio_filename"]}'.encode())
                # process(streamed)
            

        else:
            
            # message = process(streamed)
            message = "wrong code path"
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(f'{message}'.encode())

        # print(time.perf_counter() - start_time)

class CustomHTTPServer(HTTPServer):
    timeout = 60

def main():
    port = 3000
    server_address = ('', port)
    httpd = CustomHTTPServer(server_address, HelloHandler)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()

if __name__ == "__main__": 
    main()