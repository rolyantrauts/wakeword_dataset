mkdir kokoro
cd kokoro
python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate
pip3 install --upgrade pip
pip install kokoro>=0.9.4 soundfile torch torchaudio 'numpy<2' tqdm