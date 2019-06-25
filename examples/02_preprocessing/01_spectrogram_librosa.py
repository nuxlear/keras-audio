import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def main():

    audio_name = 'test.wav'
    audio_path = '../../data/audio_files/' + audio_name
    sample_rate = 44100

    # for mono audio

    wave, sr = librosa.load(audio_path, sample_rate)

    spec = librosa.stft(wave)

    # the result of STFT is complex number, so we need to get real value from it.
    # usually magnitude is used, and is converted to dB

    dB = librosa.amplitude_to_db(np.abs(spec))

    librosa.display.specshow(dB)

    # we need matplotlib to plot specshow

    plt.colorbar()
    plt.show()

    # for stereo audio

    wave, sr = librosa.load(audio_path, sample_rate, mono=False)

    specs = [librosa.stft(w) for w in wave]

    fig, axes = plt.subplots(nrows=2)

    for spec, ax in zip(specs, axes):
        spec = librosa.power_to_db(spec)    # amplitude_to_db(S) == power_to_db(S**2)
        librosa.display.specshow(spec, ax=ax)

    plt.show()


if __name__ == '__main__':
    main()
