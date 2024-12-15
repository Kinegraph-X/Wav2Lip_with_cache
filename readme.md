# A fork of the original Wav2Lip library

Optimized in perf by caching frequently used video files (and embeddings)
Also streams i/o to reduce network latency.
Run it everywhere, using the residential workers from [the other repo](https://github.com/Kinegraph-X/Wav2Lip_resident)

This server is responsible for the inference and images generation (run it on a solid GPU)
On your local machine, the workers are responsible for recording the sound form your mic, and displaying the resulting video.

### Installation

```shell
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
```

You may verify you have the right packages :

```python
import torch

if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("Cuda not available")
```

### Install Python dependancies

```shell
cd Wav2Lip_with_cache && pip install -r requirements.txt
```

### Run the server

With ngrok tunnelling and aysncio nesting (to be used from a jupyter notebook, like Google Colab)

```shell
cd Wav2Lip_with_cache && python daemon_online.py
```

Without tunnelling (appropriate for dedicated VM's and local usage)

```shell
cd Wav2Lip_with_cache && python daemon.py
```