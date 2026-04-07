https://github.com/google-research/google-research/tree/master/kws_streaming#training-on-custom-data  
Place your dataset in data2 in the above folder structure.  
Add a '_background_noise_' folder with a dummy wav file  

`source crnn_state`  

crnn.py from /models and gru.py from /layers got a few mods and works 100% (dropout...)  

ds_cnn_stream not so much and likely dur to 1.4 sec sample length where 1.5 sec would likely work much better, but not working
