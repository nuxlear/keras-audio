import librosa
import sounddevice as sd


def main():

    audio_name = 'shore.wav'
    audio_path = '../../data/audio_files/' + audio_name
    sample_rate = 44100

    wave, sr = librosa.load(audio_path, sample_rate)

    # because playing is non-blocking,
    # you must wait or program will be finished before playing audio

    sd.play(wave, sr)
    sd.wait()

    # or simply get blocked by as below

    sd.play(wave, sr, blocking=True)


if __name__ == '__main__':
    main()
