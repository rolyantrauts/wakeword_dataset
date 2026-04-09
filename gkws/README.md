https://github.com/google-research/google-research/tree/master/kws_streaming#training-on-custom-data  
Place your dataset in data2 in the above folder structure.  
Add a '_background_noise_' folder with a dummy wav file  

`source crnn_state`  

crnn.py from /models and gru.py from /layers got a few mods and works 100% (dropout...)  

ds_cnn_stream not so much and likely dur to 1.4 sec sample length where 1.5 sec would likely work much better, but not working

```
mkdir gkws
cd gkws
git clone https://github.com/google-research/google-research.git
cd google-research
git checkout fa08dcc009c73c516400dc32e13147b14196becc
cd ..
mv google-research/kws_streaming .
python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate
pip3 install --upgrade pip
pip install tensorflow-macos==2.11.0 numpy==1.26.4 tensorflow_addons tensorflow_model_optimization pydot graphviz absl-py sounddevice scipy
```
