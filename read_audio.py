import librosa
num_mfcc=13
n_fft=2048
hop_length=512
file_path = "/home/bisnu-sarkar/Data/genre_classification/Data/genres_original/blues/blues.00000.wav"
signal, sample_rate = librosa.load(file_path, sr=22050)
mfcc = librosa.feature.mfcc(y=signal,sr= 22050, n_mfcc=num_mfcc)

