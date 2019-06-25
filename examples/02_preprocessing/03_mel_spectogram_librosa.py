import librosa
import librosa.display
import matplotlib.pyplot as plt


def main():

    audio_name = 'guitar.wav'
    audio_path = '../../data/audio_files/' + audio_name
    sample_rate = 44100

    wave, sr = librosa.load(audio_path, sample_rate)

    # get mel-spectrogram from wave

    mel = librosa.feature.melspectrogram(wave, sr)

    librosa.display.specshow(librosa.power_to_db(mel))
    plt.colorbar(orientation='horizontal')
    plt.show()

    # get mel-spectrogram from pre-computed spectrogram

    spec = librosa.stft(wave)
    mel = librosa.feature.melspectrogram(S=spec)    # parameter `power` is ignored

    librosa.display.specshow(librosa.power_to_db(mel))
    plt.colorbar(orientation='horizontal')
    plt.show()


if __name__ == '__main__':
    main()
