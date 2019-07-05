import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def main():

    audio_name = 'cafe.wav'
    audio_path = '../../data/audio_files/' + audio_name
    sample_rate = 44100

    wave, sr = librosa.load(audio_path, sample_rate)

    # 1. Peak normalization

    p_norm = wave / np.max(np.abs(wave))
    print(p_norm.max(), p_norm.min())

    # 2. Loudness normalization

    dB = librosa.amplitude_to_db(wave).max()
    rms = librosa.feature.rms(wave)



    l_norm = wave

    fig, axes = plt.subplots(nrows=2)

    librosa.display.waveplot(p_norm, sr, ax=axes[0], alpha=0.5, color='r')
    librosa.display.waveplot(l_norm, sr, ax=axes[1], alpha=0.5, color='r')

    librosa.display.waveplot(wave, sr, ax=axes[0], alpha=0.5, color='c')
    librosa.display.waveplot(wave, sr, ax=axes[1], alpha=0.5, color='c')

    plt.show()


if __name__ == '__main__':
    main()
