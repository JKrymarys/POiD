import numpy as np
import cmath
from math import log, ceil
import utils

def omega(p, q):
   #The omega term in DFT and IDFT formulas
   return cmath.exp((2.0 * cmath.pi * 1j * q) / p)

def pad(lst):
   #padding the list to next nearest power of 2 as FFT implemented is radix 2
   k = 0
   while 2**k < len(lst):
      k += 1
   return np.concatenate((lst, ([0] * (2 ** k - len(lst)))))

def fft(x):
   #FFT of 1-d signals usage : X = fft(x) where input x = list containing sequences of a discrete time signals and output X = dft of x

   n = len(x)
   if n == 1:
      return x
   Feven, Fodd = fft(x[0::2]), fft(x[1::2])
   combined = [0] * n
   for m in range(int(n/2)):
       m = int(m)
       combined[m] = Feven[m] + omega(n, -m) * Fodd[m]
       combined[m + int(n/2)] = Feven[m] - omega(n, -m) * Fodd[m]
   return combined

   def ifft(X):
   #IFFT of 1-d signals usage x = ifft(X) unpadding must be done implicitly
        x = fft([x.conjugate() for x in X])
        return [x.conjugate()/len(X) for x in x]

def pad2(x):
   m, n = np.shape(x)
   M, N = 2 ** int(ceil(log(m, 2))), 2 ** int(ceil(log(n, 2)))
   F = np.zeros((M,N), dtype = x.dtype)
   F[0:m, 0:n] = x
   return F, m, n

def fft2(f):
   #FFT of 2-d signals/images with padding usage X, m, n = fft2(x), where m and n are dimensions of original signal

   f, m, n = pad2(f)
   return np.transpose(fft(np.transpose(fft(f)))), m, n

def ifft2(F, m, n):
   #IFFT of 2-d signals usage x = ifft2(X, m, n) with unpaded, where m and n are odimensions of original signal before padding

   f, M, N = fft2(np.conj(F))
   f = np.matrix(np.real(np.conj(f)))/(M*N)
   return f[0:m, 0:n]

def fftshift(F):
   # this shifts the centre of FFT of images/2-d signals
   M, N = F.shape
   R1, R2 = F[0: int(M/2), 0: int(N/2)], F[int(M/2): M, 0: int(N/2)]
   R3, R4 = F[0: int(M/2), int(N/2): N], F[int(M/2): M, int(N/2): N]
   sF = np.zeros(F.shape,dtype = F.dtype)
   sF[int(M/2): M, int(N/2): N], sF[0: int(M/2), 0: int(N/2)] = R1, R4
   sF[int(M/2): M, 0: int(N/2)], sF[0: int(M/2), int(N/2): N]= R3, R2
   return sF