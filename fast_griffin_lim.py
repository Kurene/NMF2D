import librosa
import numpy as np

def fast_griffin_lim(
        Mag, Ph, 
        n_fft, hop_length, window,
        n_iter=150, alpha=0.99
    ):
    '''
     N. Perraudin, P. Balazs and P. L. Søndergaard,
    "A fast Griffin-Lim algorithm," Proc. WASPAA2013.
    '''
    if Ph is None:
        Ph = 2*np.pi*np.random.random(Mag.phase)
    Mag_tmp = Mag.copy()
    for k in range(0, n_iter):
        if k > 0:
            X = librosa.stft(s, n_fft=n_fft, hop_length=hop_length, window=window)
            Ph = np.angle(X)
        X = Mag_tmp*np.exp(1j*Ph)
        Mag_tmp = Mag + (alpha * np.abs(X)) 
        s = librosa.istft(X, hop_length=hop_length, window=window)
        print("FastGLA",k)
    return s
    