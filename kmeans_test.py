import numpy
import time
from tslearn.clustering import TimeSeriesKMeans

def k_means_cpu(X_bt, n_clusters):
    k_means = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric='softdtw',
        max_iter=1,
        verbose=1)
    st = time.time()
    k_means.fit(X_bt)
    tt = time.time() - st
    return tt


def k_means_gpu(X_bt, n_clusters):
    k_means = TimeSeriesKMeans(n_clusters=n_clusters,
                               metric='papi',
                               max_iter=1,
                               verbose=1)
    st = time.time()
    k_means.fit(X_bt)
    tt = time.time() - st
    return tt


def similarity_test(cpu_result, gpu_result):
    pass


if __name__ == '__main__':
    X_bt = numpy.random.random((2, 1212))
    X_bt = X_bt.astype(numpy.float32)
    n_clusters = 10

    cpu_time = k_means_cpu(X_bt, n_clusters)
    print(f"cpu_time = {cpu_time}")

    try:
        gpu_time = k_means_gpu()
    except e as Exception:
        pass