!wget -q -O ngrok.zip https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip
!unzip -o ngrok.zip
!mv ngrok /usr/local/bin/ngrok
!git clone https://github.com/Kinegraph-X/Wav2Lip_with_cache
!wget -O /content/Wav2Lip_with_cache/checkpoint_path/wav2lip_gan.pth "https://huggingface.co/spaces/capstonedubtrack/Indiclanguagedubbing/resolve/416598a2eefa2f1b02bea859bda45f18208a53cb/wav2lip_gan.pth"
!wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "/content/Wav2Lip_with_cache/face_detection/detection/sfd/s3fd.pth"
!apt install ffmpeg
!pip install -r requirements.txt
!cd Wav2Lip_with_cache && python daemon_online.py