import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def main():

    audio_name = 'cafe.wav'
    audio_path = '../../data/audio_files/' + audio_name
    sample_rate = 44100

    wave, sr = librosa.load(audio_path, sample_rate)
    print('Original wave: ({:.4g}, {:.4g})'.format(wave.min(), wave.max()))

    # 1. Peak normalization

    p_norm = wave / (np.max(np.abs(wave)) + 1e-7)
    print('Peak norm:     ({:.4g}, {:.4g})'.format(p_norm.min(), p_norm.max()))

    # 2. Loudness normalization
    # The method below is RMS normalization,
    #   which can be regarded as Energy Normalization, not Loudness Normalization
    # There are several methods for Loudness Normalization, but it's dependent to purpose of Normalization
    # Thus, in this example only RMS Normalization is introduced

    rms = librosa.feature.rms(wave)

    # The target RMS value is needed, and it can be changed by methods

    max_dB = 0
    dBFS = -14

    target_dB = max_dB + dBFS
    target_amp = librosa.db_to_amplitude(target_dB)

    scale = np.sqrt(np.square(target_amp) / (np.mean(rms) + 1e-7))

    l_norm = wave * scale
    print('RMS norm:      ({:.4g}, {:.4g})'.format(l_norm.min(), l_norm.max()))

    # Visualize waves of original and normalized
    # In this example the result of RMS norm is quite similar to original sound

    fig, axes = plt.subplots(nrows=2)

    librosa.display.waveplot(p_norm[:sample_rate], sr, ax=axes[0], alpha=0.5, color='r')
    librosa.display.waveplot(l_norm[:sample_rate], sr, ax=axes[1], alpha=0.5, color='r')

    librosa.display.waveplot(wave[:sample_rate], sr, ax=axes[0], alpha=0.5, color='c')
    librosa.display.waveplot(wave[:sample_rate], sr, ax=axes[1], alpha=0.5, color='c')

    plt.show()


if __name__ == '__main__':
    main()
