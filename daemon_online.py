import asyncio
import nest_asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import ngrok
import os
import logging

from args_parser import args_parser

from process_Wav2Lip import process

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Enable nested event loops
nest_asyncio.apply()

# Set Ngrok auth token
os.environ["NGROK_AUTH_TOKEN"] = "2phoX97NqKjjsRqsAE5ijz7MSVC_2rwMGehLUbTvWEWP5YRxj"
print(os.getenv("NGROK_AUTH_TOKEN"))

logging.basicConfig(level=logging.INFO)

# Custom HTTP handler
class HelloHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        args_parser.parse(self)
        
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        
        if "path" in args_parser.params:
            with open(args_parser.params["path"], 'rb') as file: 
                self.wfile.write(file.read())
        else:
            process()
            self.wfile.write(b"Completed new processing")

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

# Main async function
async def main():
    upload_port = 5000
    server_address = ('', upload_port)
    httpd = HTTPServer(server_address, HelloHandler)
    listener = None  # Reference to the Ngrok tunnel

    try:
        # Start Ngrok tunnel
        listener = ngrok.connect(upload_port, authtoken=os.getenv("NGROK_AUTH_TOKEN"))
        print(f"Ngrok tunnel started! Public URL: {(await listener).url()}")

        # Run the HTTP server in a separate thread
        print("Starting HTTP server...")
        await asyncio.to_thread(httpd.serve_forever)

    except asyncio.CancelledError:
        print("\nServer interrupted by user.")
    finally:
        print("Shutting down HTTP server...")
        httpd.shutdown()

        print("Shutting down Ngrok tunnel...")
        # Ensure all Ngrok tunnels are forcibly stopped
        ngrok.kill()
        print("Ngrok tunnels forcibly terminated.")
        


# Run the main function in Jupyter
if __name__ == "__main__":
# Use the existing event loop
    loop = asyncio.get_event_loop()
    upload_task = loop.create_task(main())  # Start the task in the background

    try:
        loop.run_until_complete(upload_task)  # Run the event loop until the task completes
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt, shutting down gracefully.")
        tasks = asyncio.all_tasks(loop)
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