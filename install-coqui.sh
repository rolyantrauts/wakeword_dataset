git clone https://github.com/idiap/coqui-ai-TTS.git
cd coqui-ai-TTS
python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate
pip3 install --upgrade pip
pip install "coqui-tts[languages]"  tqdm torchcodec
python generate_words_coqui.py --wordlist "hey jarvis" --max_retries=5 --dest_dir=./dataset