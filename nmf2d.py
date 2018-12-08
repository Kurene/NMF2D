import numpy as np
import time


def timemeasure(func):
    def wrapper(*args, **kargs):
        start_time = time.perf_counter()
        result = func(*args, **kargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f'Proc-time: {execution_time}')
        return result
    return wrapper

class NMF2D():
    def __init__(self, n_basis, n_frames, n_pitches, n_iter,
                 init_W=None, H_sparsity=0.25):
            self.n_basis = n_basis
            self.n_frames = n_frames
            self.n_pitches = n_pitches
            self.n_iter = n_iter
            self.init_W = init_W
            self.err = [0.0 for k in range(0, n_iter)]
            self.eps = np.spacing(1)
            self.H_penalty = H_sparsity
            self.H_norm_order = 0.5

    def __init_WH(self, V):
        self.Vmax = np.max(V)
        self.Ones = np.ones(V.shape)
        self.n_row, self.n_col = V.shape
        init_H = 0.5 + 0.5*np.random.random((self.n_basis, self.n_pitches, self.n_col))
        init_W = 0.5*np.random.random((self.n_row, self.n_basis, self.n_frames))
        init_W[:,:,0] = 0.5*np.ones((self.n_row, self.n_basis))

        return init_W, init_H       
        
    def __W_regularization(self, W, order=2):
        return 0.0#np.tile(self.W_penalty*np.linspace(0, 1.0, self.n_frames)**order, (self.n_row, self.n_basis, 1))

    def __H_regularization(self, H):
        return self.H_penalty * self.__norm(H, (self.H_norm_order-2))
        
    def __update_W(self, V, W, H, order=2.0):
        VL, _ = self.__compute_VL(V, W, H)
        W_num, W_denom = np.zeros(W.shape), np.zeros(W.shape)
        W_penalty = self.__W_regularization(W)
        
        for t in range(0, self.n_frames):
            for p in range(0, self.n_pitches):
                VLp = self.__shift(VL, p, "up")
                HtpT = self.__shift(H[:,p,:], t, "right").T
                W_num[:,:,t] += np.dot(VLp, HtpT)
                W_denom[:,:,t] += np.dot(self.Ones, HtpT)
        W_new = np.clip(W*(W_num / (W_denom) + W_penalty), 0.0, self.Vmax)
        return W_new
        
    def __update_H(self, V, W, H):
        VL, _ = self.__compute_VL(V, W, H)
        H_num, H_denom = np.zeros(H.shape), np.zeros(H.shape)
        H_penalty = self.__H_regularization(H)
        
        for p in range(0, self.n_pitches):
            for t in range(0, self.n_frames):
                VLt = self.__shift(VL, t, "left")
                WtT = self.__shift(W[:,:,t], p, "down").T
                H_num[:,p,:] += np.dot(WtT, VLt)
                H_denom[:,p,:] += np.dot(WtT, self.Ones)
        H_new = np.clip(H*(H_num / (H_denom + H_penalty + self.eps)), 0.0, self.Vmax)
            
        return H_new
        
    def __norm(self, X, order):
        return np.sum(np.abs(X)**order)**(1.0/order)
        
    def __loss(self, V, W, H):
        VL, L = self.__compute_VL(V, W, H)
        Ckl = V * np.nan_to_num(np.log(VL)) - V + L
        W_reg = 0.0#self.__norm(self.__W_regularization(), 2)
        H_reg = self.H_penalty * self.__norm(H, (self.H_norm_order))
        return Ckl.sum() + W_reg + H_reg     
    
    @timemeasure
    def fit(self, V):
        W, H = self.__init_WH(V)
        for i in range(0, self.n_iter):
            W = self.__update_W(V, W, H)
            W, H = self.normalize_WH(W, H)
            H = self.__update_H(V, W, H)
            W, H = self.normalize_WH(W, H)
            self.err[i] = self.__loss(V, W, H)
            print(i+1, self.err[i])        
        self.W, self.H = W, H
        return W, H

    def __shift(self, X, n, direction):
        if n == 0:
            return X
        M, N = X.shape
        Ret = np.zeros((M,N))
        if   direction == "right":
            Ret[:,n::] = X[:,0:N-n]
        elif direction == "left":
            Ret[:,0:N-n] = X[:,n:N]
        elif direction == "down":
            Ret[n::,:] = X[0:M-n,:]
        elif direction == "up":
            #Ret[0:M-n,:] = X[n:M,:]
            Ret = np.r_[X[n:M,:],np.zeros((n,N))]
        return Ret

    def __convolution(self, W, H, factrize=False):
        V = np.zeros((self.n_row, self.n_col))
        for p in range(0, self.n_pitches):
            for t in range(0, self.n_frames):
                Wtmp = self.__shift(W[:,:,t], p, "down")
                Htmp = self.__shift(H[:,p,:], t, "right")
                V += np.dot(Wtmp, Htmp)
        return V

    def get_sources(self, W, H):
        S = np.zeros((self.n_row, self.n_col, self.n_basis))
        
        for p in range(0, self.n_pitches):
            for t in range(0, self.n_frames):
                Wtmp = self.__shift(W[:,:,t], p, "down")
                Htmp = self.__shift(H[:,p,:], t, "right")
                for k in range(0, self.n_basis):
                    S[:,:,k] += np.outer(Wtmp[:,k], Htmp[k,:])
        return S        
        
    def __compute_VL(self, V, W, H, eps=np.spacing(1)):
        L = self.__convolution(W, H)
        VL = np.nan_to_num(V/L)
        return VL, L

    def normalize_WH(self, W, H, return_2d=False):
        W2d = np.reshape(W, (self.n_row, self.n_basis*self.n_frames))
        H2d = np.reshape(H, (self.n_basis*self.n_pitches, self.n_col))
        
        for k in range(0, self.n_basis):
            fact = np.sum(W2d[:,k])
            W2d[:,k] /= fact
            H2d[k,:] *= fact
        
        if return_2d:
            return W2d, H2d
        else:
            W = np.reshape(W2d, (self.n_row, self.n_basis, self.n_frames))
            H = np.reshape(H2d, (self.n_basis, self.n_pitches, self.n_col))
            return W, H

    def reconstruct(self, W, H):
        return self.__convolution(W, H)
    