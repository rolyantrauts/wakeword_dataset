git clone https://github.com/k2-fsa/sherpa-onnx.git
cd sherpa-onnx
python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate
pip3 install --upgrade pip
pip install sherpa-onnx sherpa-onnx-bin torch torchaudio 'numpy<2' tqdm  onnxruntime soundfile
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2
tar xvf vits-piper-en_US-libritts_r-medium.tar.bz2
rm vits-piper-en_US-libritts_r-medium.tar.bz2