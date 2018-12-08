import librosa

path = "audio.mp3"
sr = 44100
y, sr = librosa.load(path, sr=sr)
C = librosa.cqt(y=y, sr=sr)
print(C)
y_hat = librosa.icqt(C=C, sr=sr)
librosa.output.write_wav('reconst.wav', y_hat, sr)
        