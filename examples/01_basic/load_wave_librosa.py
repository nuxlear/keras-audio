import librosa
import timeit


def main():

    audio_name = 'shore.wav'
    audio_path = '../../data/audio_files/' + audio_name
    sample_rate = 44100     # usually use 16000, 22050, 44100, 48000, etc.
    is_stereo = True        # if wave file is stereo

    start = timeit.default_timer()
    wave, sr = librosa.load(audio_path, sample_rate, mono=not is_stereo)
    stop = timeit.default_timer()

    print('Successfully loaded wave: {}'.format(audio_name))
    print('Elapsed time: {:.4g}s'.format(stop - start))
    print('shape: {}'.format(wave.shape))       # loaded wave in librosa is Numpy array


if __name__ == '__main__':
    main()
