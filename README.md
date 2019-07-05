# keras-audio

Implementation and examples for audio signal processing with using Keras. 

This repository is supplementary for my presentation: 
["Hand-On Audio for Deep Learning (Practice)"](https://docs.google.com/presentation/d/1SsesME3qtCvvJy6yqW1EOXPC0Rc-vwb0OmS4QXi_cao/edit#slide=id.g5cf03feef5_0_860) 




### Sample audio

In ```data/audio_files```, 10 audio samples from [Freesound](https://freesound.org/). 


### Example codes

There are some example codes for treating audio data. 
It basically uses ```librosa``` for I/O, ```sounddevice``` and ```IPython.display.Audio``` for playback.
In this repo, all example codes will use them by default, 
except for basic codes for other libraries. 

You can choose another options like as below:

for I/O:

- [wave (Python3)](https://docs.python.org/3/library/wave.html)
- [scipy.io](https://docs.scipy.org/doc/scipy/reference/io.html)
- [pydub](https://docs.python.org/3/library/wave.html)

for playback:

- [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/docs/)
- [scikit-sound](http://work.thaslwanter.at/sksound/html/)
- [pygame.mixer](https://www.pygame.org/docs/ref/mixer.html)


#### 01. Basic

Example codes for basic load/save and play to listen. 

- load/save wave
- playback
- plot waveform

#### 02. Preprocessing

Example codes for feature extraction, converting, and feature visualization.

- STFT (spectrogram)
- magnitude & phase
- mel-spectrogram
- MFCC (mel-frequency cepstral coefficient)
- inverse STFT
- RMS normalization

#### 03. Augmentation

