from __future__ import absolute_import
from __future__ import print_function

import numpy
import time

from GPUDTW import cuda_dtw
from GPUDTW import cpu_dtw, dtw_1D_jit2

if __name__ == '__main__':
    S = numpy.random.random ((3,1024))
    S = S.astype(numpy.float32)
    T = numpy.random.random ((1312,1024))
    T = T.astype(numpy.float32)

    # t0 = time.time()
    # ret_cpu =cpu_dtw (S, T, dtw_1D_jit2)
    # print ("cpu time",time.time()-t0)

    t0 = time.time()
    ret_cuda = cuda_dtw (S, T)
    print(ret_cuda)
    print(ret_cuda.shape)
    print ("cuda time:",time.time()-t0)
    # cuda_verify = numpy.sqrt((ret_cuda - ret_cpu)**2)
    # print ("Maximum Deviation in cuda with CPU ", cuda_verify.max())