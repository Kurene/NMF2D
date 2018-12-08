import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
from nmf2d import NMF2D
from fast_griffin_lim import fast_griffin_lim

def plotspec(V, W, H, L, y_axis='cqt_note'):
    plt.clf()
    plt.subplot(2,2,1)
    librosa.display.specshow(librosa.amplitude_to_db(V, ref=np.max),
                             y_axis=y_axis, x_axis='time', cmap='jet')

    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(2,2,2)
    if W.shape[1]==2:
        W = W[::-1,:]       
    librosa.display.specshow(W, y_axis=y_axis, cmap='jet')
    plt.title('Reconst. spec.')
    plt.colorbar()

    plt.subplot(2,2,3)
    librosa.display.specshow(librosa.amplitude_to_db(L, ref=np.max),
                             y_axis=y_axis, x_axis='time', cmap='jet')
    plt.title('Reconst. spec.')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(2,2,4)
    librosa.display.specshow(H, x_axis='time', cmap='jet')
    plt.title('Activation')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
      

if __name__ == "__main__":
    path = sys.argv[1]
    n_basis = int(sys.argv[2])
    n_frames = int(sys.argv[3])
    n_pitches = int(sys.argv[4])
    n_iter = int(sys.argv[5])
    sr = 16000
    n_fft = 1024
    hop_length = 512
    window = "hann"
    offset = 0
    duration = 30
    n_bins = 84
    y_axis = "cqt_note"

    y, sr = librosa.core.load(path, sr=16000, offset=offset, duration=duration, mono=True)
    #X = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    X = librosa.cqt(y=y, sr=sr, hop_length=hop_length)
    X_abs, X_phase = np.abs(X), np.angle(X)
    X_gain = np.max(X_abs)
    X_abs /= X_gain
    
    model = NMF2D(n_basis, n_frames, n_pitches, n_iter)
    W, H = model.fit(X_abs)
    W2d, H2d = model.normalize_WH(W, H, return_2d=True)
    Y = model.reconstruct(W, H) 
    plotspec(X_abs, W2d, H2d, Y, y_axis=y_axis)
    
    # Sources reconstructed with W and H
    Y = model.get_sources(W, H) 
    for k in range(0, n_basis):
        plotspec(X_abs, W[:,k,:], H[k,:,:], Y[:,:,k], y_axis=y_axis)
        #y_est = fast_griffin_lim(Y[:,:,k], np.angle(X), n_fft=n_fft, hop_length=hop_length, window=window)
        #librosa.output.write_wav('reconst_'+str(k)+'.wav', y_est, sr)
        
    