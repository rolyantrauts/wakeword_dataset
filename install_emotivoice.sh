git clone https://github.com/netease-youdao/EmotiVoice.git
cd EmotiVoice
conda create -n EmotiVoice python=3.8 -y
conda activate EmotiVoice
pip install torch torchaudio
pip install numpy numba scipy transformers soundfile yacs g2p_en jieba pypinyin pypinyin_dict
python -m nltk.downloader "averaged_perceptron_tagger_eng"

git lfs install
git clone https://www.modelscope.cn/syq163/WangZeJun.git
git clone https://www.modelscope.cn/syq163/outputs.git
