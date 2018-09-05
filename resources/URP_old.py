# -*- coding: utf-8 -*-

import numpy as np

from scipy.optimize    import curve_fit
from scipy.interpolate import interp1d

#=============================================================================

def periodogram(X, fs):
    """
    Estimates the one-side power spectrum of a signal.
 
    Parameters:  X:   time series
                 fs:  sampling frequency
    """
    
    X   =  X.ravel()
    N   =  len(X)
    M   =  np.int(N/2) + 1 
    fs  =  np.float(fs)
    
    Sx  =  np.fft.fft(X)
    Sx  =  np.real(Sx*Sx.conj())*2/N/fs
                
    return np.linspace(0,fs/2,M), Sx[0:M]


#=============================================================================

def simulate(Sx, fs):
    """
    Simulate a signal from given spectral density.
 
    Parameters:  Sx:  one-side power spectrum
                 fs:  sampling frequency
    """
    
    Sx  =  Sx.ravel()
    M   =  len(Sx)
    N   =  2*(M - 1)
    fs  =  np.float(fs)
        
    Sx  =  N*fs*Sx/2
    
    phi    =  2*np.pi*np.random.rand(M)
    phi[0] =  0. # no phase for mean value!!!
    
    Pw  =  np.sqrt(Sx)*(np.cos(phi) + 1j*np.sin(phi))
    Pw  =  np.hstack((Pw,np.conj(Pw[-2:0:-1])))
 
    return np.linspace(0,N/fs,N), np.real(np.fft.ifft(Pw))


#=============================================================================

def cx2sx(Cx, fs):
    """
    Converts autocorrelation function to one-side spectral density.
 
    Parameters:  Cx:  autocorrelation function
                 fs:  sampling frequency
    """

    Cx  =  Cx.ravel()
    M   =  len(Cx)
    fs  =  np.float(fs)
    
    Cx  =  np.hstack((Cx, Cx[-2:0:-1]))
    Sx  =  np.fft.fft(Cx)*2/fs

        
    return np.linspace(0,fs/2,M), np.real(Sx[0:M])


#=============================================================================

def sx2cx(Sx, fs):
    """
    Converts one-side spectral density to autocorrelation function.
 
    Parameters:  Sx: (one-sided) spectral density
                 f:   frequency (independent variable)
    """
    
    Sx  =  Sx.ravel()
    M   =  len(Sx)
    fs  =  np.float(fs)
    
    Sx  =  np.hstack((Sx, Sx[-2:0:-1]))
    Cx  =  np.fft.ifft(Sx)*fs/2


    return np.linspace(0,M/fs,M), np.real(Cx[0:M])

#=============================================================================

def mov_average(X, n=3):
    '''
    Simple moving average filter with a square window.
    
    Parameters:  X: time series
                 n: window width (will be forced to an odd number)
    '''
    
    X =  X.ravel()
    N =  len(X)
    n =  np.int(np.round(n))     # round to nearest integer
    n =  n - (1 - np.mod(n,2))   # n is odd or will be decreased by 1
    m =  np.int((n -  1)/2)      # half window
    
    Y =  np.empty(N)
    
    for i in range(N):
        
        i0   = i - m      if i > m     else 0
        i1   = i + m + 1  if i < N - 1 else N
        
        Y[i] = X[i0:i1].mean()
    
    return Y


#=============================================================================

def bandpass(X, fs, band):
    """
    Absolute bandpass filter. Series size is doubled by trailing zeros
    before filtering, in order to avoid aliasing.
 
    Parameters:  X:    time series
                 fs:   sampling frequency
                 band: frequency band to keep as a list: [f_low, f_high]
    """
    
    X    =  X.ravel()
    N    =  len(X)
    X    =  np.vstack((X, np.zeros(N))).ravel()
    
    M    =  N + 1
    fs   =  np.float(fs)
    f    =  np.linspace(0,fs/2,M)
    Xw   =  np.fft.fft(X)[0:M]

    Xw[(f <= band[0]) | (f > band[1])] = 0.

    X    =  np.real(np.fft.ifft(np.hstack((Xw, np.conj(Xw[-2:0:-1])))))

    return X[0:N]


#=============================================================================

def davemax(X, fs, Td):
    """
    Peak factor from Davenport's formula.
 
    Parameters:  X:    time series
                 fs:   sampling frequency
                 Td:   time span for peak factor
    """
    
    e  = 0.5772156649
    
    f, Sx = periodogram(X - X.mean(), fs)
    
    df = f[1] - f[0]
    
    m0 = np.trapz(Sx,     dx=df)
    m2 = np.trapz(Sx*f*f, dx=df)
    
    nu = Td*np.sqrt(m2/m0)
    if (nu < 1): nu = 1

    Lg = np.sqrt(2*np.log(nu))
    if (Lg < np.sqrt(e)): Lg = np.sqrt(e)
    
    return Lg + e/Lg


#=============================================================================

def splitmax(X, fs, Td):
    """
    Uses the splitmax method to estimate the maximum absolute value of X
    for an observation time Td. If X is a standard gaussian process, this
    maximum corresponds to the peak factor, g, for Td.
 
    Parameters:  X:    time series
                 fs:   sampling frequency
                 Td:   time span for peak factor
    """

#-----------------------------------------------------------------------------

    def split(X):
    
        X1 = X[0::2]
        X2 = X[1::2]
    
        if not len(X1): 
            return np.array([])
    
        if len(X1) > len(X2):
            X1 = X1[:-1]
    
        return np.max(np.vstack((X1, X2)), axis=0)
        
#-----------------------------------------------------------------------------

    n    =  len(X)
    Y    =  split(np.abs(X))
    nmax =  np.array([])
    Xmax =  np.array([])

    while np.size(Y):
        nmax = np.append(nmax,n/len(Y))
        Xmax = np.append(Xmax,Y.mean())
        Y    = split(Y)
    
    f = interp1d(np.log(nmax), Xmax, kind='quadratic')
    
    return nmax, Xmax, f(np.log(Td*fs))


#=============================================================================

def integrate(X, fs, band):
    """
    Frequency domain integration of a time series.
 
    Parameters:  X:    time series
                 fs:   sampling frequency
                 band: frequency band to keep, tuple: [f_low, f_high)
    """
    
    X    =  X.ravel()
    N    =  len(X)
    M    =  np.int(N/2) + 1
    fs   =  np.float(fs)
    f    =  np.linspace(0,fs/2,M)
    Xw   =  np.fft.fft(X)[0:M]
    f[0] =  1.
    Xw   =  Xw / (2j*np.pi*f) 
    
    Xw[(f <= band[0]) | (f > band[1])] = 0.

    return np.real(np.fft.ifft(np.hstack((Xw, np.conj(Xw[-2:0:-1])))))


#=============================================================================

def differentiate(X, fs, band):
    """
    Frequency domain differentiation of a time series.
 
    Parameters:  X:    time series
                 fs:   sampling frequency
                 band: frequency band to keep, tuple: [f_low, f_high)
    """
    
    X    =  X.ravel()
    N    =  len(X)
    M    =  np.int(N/2) + 1
    fs   =  np.float(fs)
    f    =  np.linspace(0,fs/2,M)
    Xw   =  np.fft.fft(X)[0:M]

    Xw   =  Xw * (2j*np.pi*f)
    
    Xw[(f <= band[0]) | (f > band[1])] = 0.

    return np.real(np.fft.ifft(np.hstack((Xw, np.conj(Xw[-2:0:-1])))))


#=============================================================================

def resample(t, X, Td, fs):
    """
    Resampling irregular time step to fixed time step
 
    Parameters:  t:    irregular time
                 X:    time series
                 Td:   time limit to retain
                 fs:   resampling frequency
    """
    
    if (Td > t[-1]): 
        Td = t[-1]
     
    Ni   =  np.int(fs*Td)
    ti   =  np.linspace(0,Td,Ni)
    rF   =  interp1d(t, X, kind='quadratic')

    return ti, rF(ti)


#=============================================================================

def EQ_Duhamel(wn, zt, fs, u0, v0, ac):
    """
    Integrates the dynamic equilibrium differential equation by Duhamel.

    Parameters:  wn:  natural frequency (rad/s)
                 zt:  damping  (nondim)
                 fs:  sampling frequency of excitation
                 u0:  initial position
                 v0:  initial velocity
                 ac:  external force divided by system mass (m/s^2)
    """

    ac  =  ac.ravel()
    N   =  len(ac)
    fs  =  np.float(fs)
    
    wd  =  wn*np.sqrt(1 - zt**2)
    t   =  np.linspace(0,N/fs,N)
    dt  =  1/fs

    et  =  np.exp(zt*wn*t)
    st  =  np.sin(wd*t)
    ct  =  np.cos(wd*t)

    u   = (u0*ct + (v0 + u0*zt*wn)*st/wd)/et

    A   = (ac*et*ct)*dt;   A = np.cumsum(A)
    B   = (ac*et*st)*dt;   B = np.cumsum(B)

    u   =  u + (A*st - B*ct)/et/wd

    
    return t, u


#=============================================================================

def EQ_Fourier(wn, zt, fs, ac):
    """
    Integrates the dynamic equilibrium differential equation by Fourier.

    Parameters:  wn:  natural frequency (rad/s)
                 zt:  damping  (nondim)
                 fs:  sampling frequency of excitation
                 ac:  external force divided by system mass (m/s^2)
    """

    ac  =  ac.ravel()
    N   =  len(ac)
    M   =  N/2 + 1
    fs  =  np.float(fs)

    dt  =  1.0/fs
    K   =  wn*wn

    w   =  np.linspace(0.,fs*np.pi,M)/wn
    t   =  np.linspace(0.,N*dt,N)

    Hw  = (K*((1.0 - w**2) + 1j*(2*zt*w)))**(-1)
    Hw  =  np.hstack((Hw,np.conj(Hw[-2:0:-1])))

    u   =  np.real(np.fft.ifft(Hw*np.fft.fft(ac)))
    
    return t, u


#=============================================================================

def random_decrement(X, fs, ref, Xth, n):
    """
    Estimate a free decay response of a dynamic system from the system
    response to a wide band excitation

    Parameters:  X:   time series (matrix, each row is an output)
                 ref: row of X to be used as reference signal
                 Xth: threshold level that defines upcrossing level
                 n:   length of free decay response
    """

    if (len(X.shape) == 1): 
        X = np.expand_dims(X, axis=0) 

    Nsign =  X.shape[0]
    N     =  X.shape[1]
    kup   = (X[ref, 0:(N-1)] < Xth) & (X[ref, 1:N] >= Xth) 
    kdown = (X[ref, 0:(N-1)] > Xth) & (X[ref, 1:N] <= Xth) 

    Xrd   =  np.zeros((Nsign, n))
    nrd   =  0
    
    for kk in range(Nsign):
        for ii in range(N-n):
            if kup[ii] | kdown[ii]:
                Xrd += X[:,ii:(ii+n)]
                nrd += 1
    
    return np.linspace(0, n/fs, n), Xrd/nrd


#=============================================================================

def fit_decay(X, fs):
    """
    Fit the theoretical free decay function of a dynamic system to the
    experimental measurement "X".

    Parameters:  X:   time series
                 fs:  sampling frequency of excitation
    """
#-----------------------------------------------------------------------------
    def decay(t, Xm, Xp, fn, zt, ph):
        
        wn =  2*np.pi*fn
        wd =  wn*np.sqrt(1. - zt*zt)
        
        return Xm + Xp*np.exp(-zt*wn*t)*np.sin(wd*t - ph)
        
#-----------------------------------------------------------------------------

    N    =  len(X)
    fs   =  np.float(fs)
    t    =  np.linspace(0, N/fs, N)
    
    Xm   =  X.mean()

    f, S =  periodogram(X-Xm, fs)
    fn   =  f[np.argmax(S)]
    p0   =  np.array([X.mean(), X.max(), fn, 0.01, -np.pi/2])

    try:                
        p, cv =  curve_fit(decay, t, X, p0=p0)
    except:
        p     =  np.zeros(5)
        print('Not able to fit decay function!!!')
        pass
    
    Xm    = p[0]
    Xp    = p[1]
    fn    = p[2]
    zt    = p[3]
    phase = p[4]

    return Xm, Xp, fn, zt, phase, decay(t, Xm, Xp, fn, zt, phase)

#=============================================================================
