import librosa
import numpy as np
import sounddevice as sd


def main():

    sr = 44100

    time = 2.
    freq = 440

    # create continuous sine wave with frequency

    samples = np.arange(time * sr) / sr
    wave = np.sin(2 * np.pi * freq * samples)

    # listen and check generated wave

    sd.play(wave, blocking=True)

    audio_name = 'test.wav'
    audio_path = '../../data/audio_files/' + audio_name

    librosa.output.write_wav(audio_path, wave, sr)


if __name__ == '__main__':
    main()
