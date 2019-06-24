import librosa
import matplotlib.pyplot as plt
import numpy as np


def main():

    audio_name = 'shore.wav'
    audio_path = '../../data/audio_files/' + audio_name
    sample_rate = 44100

    # for mono audio

    wave, sr = librosa.load(audio_path, sample_rate)
    time = np.arange(len(wave)) / sr

    plt.plot(time, wave)
    plt.xlabel('time (s)')
    plt.ylabel('amplitude')
    plt.show()

    # for stereo audio
    # NOTE: the shape of `wave` will be (2, len)

    wave, sr = librosa.load(audio_path, sample_rate, mono=False)
    time = np.arange(wave.shape[-1]) / sr

    fig, axes = plt.subplots(nrows=2)

    for wav, ax in zip(wave, axes):
        ax.plot(time, wav)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('amplitude')

    plt.show()


if __name__ == '__main__':
    main()
