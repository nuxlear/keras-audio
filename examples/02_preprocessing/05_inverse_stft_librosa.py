import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


def main():

    sample_rate = 44100

    n_fft = 512
    n_frame = 1000

    # create arbitrary spectrogram as Numpy array
    # for convenience, create spectrogram in range of dB and convert to power

    spec = np.random.normal(-20, 2, [1 + n_fft // 2, n_frame])
    spec[20:30] += 50

    spec = librosa.db_to_amplitude(spec)

    # plot created spectrogram

    librosa.display.specshow(librosa.amplitude_to_db(spec))
    plt.colorbar()
    plt.show()

    # inverse STFT and playback

    wave = librosa.istft(spec)
    print('wave shape: {}'.format(wave.shape))

    sd.play(wave, samplerate=sample_rate, blocking=True)


if __name__ == '__main__':
    main()
