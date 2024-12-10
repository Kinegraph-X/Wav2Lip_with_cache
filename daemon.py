from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse as parse
from args_parser import args_parser

from process_Wav2Lip import process





import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Custom HTTP handler
class HelloHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        args_parser.parse(self)
        
        process()

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"Completed new processing")
        # """

    def do_POST(self):
        
        args_parser.parse(self)
        if not args_parser.params["audio_filename"]:
            print("Audio file not received, aborting...")
            self.send_response(500)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"Audio file not received, aborting...")
            return

        process()

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"Completed new processing")

def main():
    port = 3000
    server_address = ('', port)
    httpd = HTTPServer(server_address, HelloHandler)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()

if __name__ == "__main__": 
    main()