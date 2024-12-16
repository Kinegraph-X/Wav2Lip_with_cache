# ngrok, in case you need it (only for jupyter notebooks)
wget -q -O ngrok.zip https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip
unzip -o ngrok.zip
mv ngrok /usr/local/bin/ngrok

# Normally, you've already cloned this repo
git clone https://github.com/Kinegraph-X/Wav2Lip_with_cache

# Maybe a few days later, if you've already cloned the repo
cd Wav2Lip_with_cache && git fetch
cd Wav2Lip_with_cache && git pull

# Download the files too big to fit on Github
wget -O /content/Wav2Lip_with_cache/checkpoint_path/wav2lip_gan.pth "https://huggingface.co/spaces/capstonedubtrack/Indiclanguagedubbing/resolve/416598a2eefa2f1b02bea859bda45f18208a53cb/wav2lip_gan.pth"

wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "/content/Wav2Lip_with_cache/face_detection/detection/sfd/s3fd.pth"

# I'm providing a version of the cached embeddings corresponding to the video in the repo (this speeds up the first run)
mkdir /content/Wav2Lip_with_cache/cache
mkdir /content/Wav2Lip_with_cache/cache/face_detection
wget "http://fluoman.net/kinegraphx_avatar/cache/face_detection/4adfbccd5f9577dbd9fa024cac6fd5fb_face_detection_master.npy" -O "/content/Wav2Lip_with_cache/cache/face_detection/4adfbccd5f9577dbd9fa024cac6fd5fb_face_detection_master.npy"

# Maybe you don't have it installed... Yet...
apt install ffmpeg