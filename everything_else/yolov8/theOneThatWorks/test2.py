import timeit
import numpy as np
import cv2 as cv

npTmp = np.random.random((1024, 1024)).astype(np.float32)

npMat1 = np.stack([npTmp, npTmp], axis=2)
npMat2 = npMat1

cuMat1 = cv.cuda_GpuMat()
cuMat2 = cv.cuda_GpuMat()
cuMat1.upload(npMat1)
cuMat2.upload(npMat2)

# Define cuMat1 and cuMat2 before using them in the lambda function
time_gpu = timeit.timeit(lambda: cv.cuda.gemm(cuMat1, cuMat2, 1, None, 0, None, 1), number=1)
time_cpu = timeit.timeit(lambda: cv.gemm(npMat1, npMat2, 1, None, 0, None, 1), number=1)

print(f"GPU time: {time_gpu} seconds")
print(f"CPU time: {time_cpu} seconds")
