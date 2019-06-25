import librosa
import librosa.display
import matplotlib.pyplot as plt


def main():

    audio_name = 'harp.wav'
    audio_path = '../../data/audio_files/' + audio_name
    sample_rate = 44100

    wave, sr = librosa.load(audio_path, sample_rate)

    # get mfcc from wave

    mfcc = librosa.feature.mfcc(wave, sr)

    librosa.display.specshow(mfcc)
    plt.colorbar()
    plt.show()

    # get mel-spectrogram from pre-computed mel-spectrogram

    mel = librosa.feature.melspectrogram(wave, sr)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel))

    librosa.display.specshow(mfcc)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
