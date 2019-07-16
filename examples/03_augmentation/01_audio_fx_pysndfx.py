import librosa
from pysndfx import AudioEffectsChain
import sounddevice as sd


def main():

    audio_name = 'rhythm.wav'
    audio_path = '../../data/audio_files/' + audio_name
    sample_rate = 44100

    wave, sr = librosa.load(audio_path, sample_rate)

    fx = (
        AudioEffectsChain()
        .highshelf()
        .delay()
        .phaser()
        .reverb()
        .lowshelf()
    )

    result = fx(wave)

    print('Playing original sound...')
    sd.play(wave, sample_rate, blocking=True)
    print('Playing Effected sound...')
    sd.play(result, sample_rate, blocking=True)


if __name__ == '__main__':
    main()
