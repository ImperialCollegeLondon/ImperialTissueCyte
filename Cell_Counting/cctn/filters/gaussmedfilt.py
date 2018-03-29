import numpy
import math
import scipy.stats
from im2col import im2col
from col2im import col2im

def gausskern(SHAPE,SIGMA):
    m,n = [(ss-1.)/2. for ss in SHAPE]
    y,x = numpy.ogrid[-m:m+1,-n:n+1]
    h = numpy.exp( -(x*x + y*y) / (2.*SIGMA*SIGMA) )
    h[ h < numpy.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gaussmedfilt(A,RADIUS,SIGMA):
    N = (RADIUS*2)+1

    G = gausskern((N,N),SIGMA)
    G_norm = G/numpy.amax(G)
    g = gausskern((N,N),SIGMA/1.5)
    g_norm = g/numpy.amax(g)
    Gg = G_norm-g_norm
    Gg_norm = Gg/numpy.amax(Gg)
    Gg_col = Gg_norm.reshape((numpy.size(Gg_norm),1))

    A_pad = numpy.lib.pad(A,((int(math.floor(N/2.)),int(math.floor(N/2.))),(int(math.floor(N/2.)),int(math.floor(N/2.)))),'constant')
    A_col = im2col(A_pad, (N, N))

    A_weighted = numpy.multiply(A_col,Gg_col)

    A_sort = numpy.sort(A_weighted,0)

    A_median = A_sort[int(math.floor(N*N/2)+1),:]

    A_im = col2im(A_median, (N, N), numpy.shape(A_pad))

    return A_im
