import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def main():

    audio_name = 'cafe.wav'
    audio_path = '../../data/audio_files/' + audio_name
    sample_rate = 44100

    wave, sr = librosa.load(audio_path, sample_rate)

    mel = librosa.feature.melspectrogram(wave, sr, power=1)

    # plot original mel-spectrogram

    librosa.display.specshow(librosa.power_to_db(mel))
    plt.colorbar(orientation='horizontal')
    plt.show()

    n_freq, n_time = mel.shape
    m = np.mean(mel)

    # augment method 1: warping
    # in the paper, it said that this method is too expensive to influence augmentation.
    # thus, in this examples the method is skipped

    # augment method 2: masking
    # masking for frequency domain, and time domain

    mask_freq_size = np.random.randint(0.2 * n_freq)
    mask_time_size = np.random.randint(0.2 * n_time)

    mask_freq_pos = np.random.randint(0, n_freq - mask_freq_size)
    mask_time_pos = np.random.randint(0, n_time - mask_time_size)

    mel[mask_freq_pos: mask_freq_pos + mask_freq_size] = m
    mel[:, mask_time_pos: mask_time_pos + mask_time_size] = m

    # plot mel-spectrogram to check the result

    librosa.display.specshow(librosa.power_to_db(mel))
    plt.colorbar(orientation='horizontal')
    plt.show()


if __name__ == '__main__':
    main()
