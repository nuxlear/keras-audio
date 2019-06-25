import librosa
import librosa.display
import matplotlib.pyplot as plt


def main():

    audio_name = 'guitar.wav'
    audio_path = '../../data/audio_files/' + audio_name
    sample_rate = 44100

    wave, sr = librosa.load(audio_path, sample_rate)

    mel = librosa.feature.melspectrogram(wave, sr)

    librosa.display.specshow(librosa.power_to_db(mel))
    plt.show()


if __name__ == '__main__':
    main()
