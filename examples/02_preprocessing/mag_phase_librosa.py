import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def main():

    audio_name = 'cafe.wav'
    audio_path = '../../data/audio_files/' + audio_name
    sample_rate = 44100

    wave, sr = librosa.load(audio_path, sample_rate)

    spec = librosa.stft(wave)

    # to get magnitude & phase, get absolute & angle of complex value

    mag = np.abs(spec)
    phase = np.angle(spec)

    # plot magnitude & phase

    librosa.display.specshow(librosa.power_to_db(mag))
    plt.show()

    librosa.display.specshow(phase)
    plt.show()

    # or you can simply use librosa function

    # NOTE: the phase from the function below is form of `e^(1.0j * angle)`,
    # so phase will be complex number, which means (spec) = (mag) * (phase)
    # if you need to get real number of phase, use method above
    # please read for more info:
    #       https://librosa.github.io/librosa/generated/librosa.core.magphase.html

    mag, phase = librosa.magphase(spec)

    librosa.display.specshow(librosa.power_to_db(mag))
    plt.show()

    librosa.display.specshow(phase)     # it just uses magnitude of phase from librosa.magphase()
    plt.show()


if __name__ == '__main__':
    main()
