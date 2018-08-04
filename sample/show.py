import os

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from sample.read import readImage

def showImage(npImg):
    n = npImg.shape[0]
    fig = plt.figure(figsize=(30, 30))
    cols = 8
    rows = n / cols + 1
    for i in range(1, n + 1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(npImg[i - 1], cmap='gray', vmin=0, vmax=1)
    plt.show()

def getMinMax(npImg):
    inf = 987654321
    maxv = -inf
    minv = inf
    for d in range(npImg.shape[0]):
        for h in range(npImg.shape[1]):
            for w in range(npImg.shape[2]):
                maxv = max(maxv, npImg[d, h, w])
                minv = min(minv, npImg[d, h, w])
    return minv, maxv

"""
basePath = os.path.abspath('..')
trainPath = os.path.join(basePath, 'data/train')

sitkImg = readImage(os.path.join(trainPath, 'Case00.mhd'))
sitkGT = readImage(os.path.join(trainPath, 'Case00_segmentation.mhd'))
print('sitkImg: {}'.format(sitkImg.GetSize()))
print('sitkGT: {}'.format(sitkGT.GetSize()))

# 1. Represent an image and a ground-truth directly without any preprocessing
npImg = sitk.GetArrayFromImage(sitkImg)
npGT = sitk.GetArrayFromImage(sitkGT)
print('first step:')
minv, maxv = getMinMax(npImg)
print('in npImg values of max and min are: {}, {}'.format(maxv, minv))
minv, maxv = getMinMax(npGT)
print('in npGT values of max and min are: {}, {}'.format(maxv, minv))
showImage(npImg)
showImage(npGT)

# 2. Represent an image and a ground-truth with rescaling supproted by SimpleITK
rescalFilt = sitk.RescaleIntensityImageFilter()
rescalFilt.SetOutputMaximum(255)
rescalFilt.SetOutputMinimum(0)
rescaledImg = rescalFilt.Execute(sitkImg)
rescaledGT = rescalFilt.Execute(sitkGT)
npImg = sitk.GetArrayFromImage(rescaledImg)
npGT = sitk.GetArrayFromImage(rescaledGT)
print('second step:')
minv, maxv = getMinMax(npImg)
print('in npImg values of max and min are {}, {}'.format(maxv, minv))
minv, maxv = getMinMax(npGT)
print('in npGT values of max and min are {}, {}'.format(maxv, minv))
showImage(npImg)
showImage(npGT)

# 3.    Represent an image and a ground-truth with all preprocessing supported by SimpleITK
#       Set to rescaling values are 0 and 1
rescalFilt = sitk.RescaleIntensityImageFilter()
rescalFilt.SetOutputMaximum(1)
rescalFilt.SetOutputMinimum(0)
rescaledImg = rescalFilt.Execute(sitkImg)
rescaledGT = rescalFilt.Execute(sitkGT)
#
factor = np.asarray(sitkImg.GetSpacing()) / [1.0, 1.0, 1.5]
factorSize = np.asarray(sitkImg.GetSize() * factor, dtype=float)
newSize = np.max([factorSize, [128, 128, 64]], axis=0).astype(dtype=np.uint32).tolist()
#
T = sitk.AffineTransform(3)
T.SetMatrix(sitkImg.GetDirection())
#
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(rescaledImg)
resampler.SetOutputSpacing([1.0, 1.0, 1.5])
resampler.SetSize(newSize)
resampler.SetInterpolator(sitk.sitkLinear)
resampledImg = resampler.Execute(rescaledImg)
resampledGT = resampler.Execute(rescaledGT)
imgCentroid = np.asarray(newSize, dtype=np.float32) / 2.0
imgStartPx = (imgCentroid - np.asarray([128, 128, 64], dtype=np.float32) / 2.0).astype(dtype=np.int32).tolist()
#
regionExtractor = sitk.RegionOfInterestImageFilter()
regionExtractor.SetSize([128, 128, 64])
regionExtractor.SetIndex(imgStartPx)
croppedImg = regionExtractor.Execute(resampledImg)
croppedGT = regionExtractor.Execute(resampledGT)
npImg = sitk.GetArrayFromImage(croppedImg)
npGT = sitk.GetArrayFromImage(croppedGT)
print('third step:')
minv, maxv = getMinMax(npImg)
print('in npImg values of max and min are {}, {}'.format(maxv, minv))
minv, maxv = getMinMax(npGT)
print('in npGT values of max and min are {}, {}'.format(maxv, minv))
showImage(npImg)
showImage(npGT)

# 4.    Represent an image and ground-truth with all preprocessing supported by SimpleITK
#       Set to rescaling values are 0 and 255
rescalFilt = sitk.RescaleIntensityImageFilter()
rescalFilt.SetOutputMaximum(255)
rescalFilt.SetOutputMinimum(0)
rescaledImg = rescalFilt.Execute(sitkImg)
rescaledGT = rescalFilt.Execute(sitkGT)
#
factor = np.asarray(sitkImg.GetSpacing()) / [1.0, 1.0, 1.5]
factorSize = np.asarray(sitkImg.GetSize() * factor, dtype=float)
newSize = np.max([factorSize, [128, 128, 64]], axis=0).astype(dtype=np.uint32).tolist()
#
T = sitk.AffineTransform(3)
T.SetMatrix(sitkImg.GetDirection())
#
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(rescaledImg)
resampler.SetOutputSpacing([1.0, 1.0, 1.5])
resampler.SetSize(newSize)
resampler.SetInterpolator(sitk.sitkLinear)
resampledImg = resampler.Execute(rescaledImg)
resampledGT = resampler.Execute(rescaledGT)
imgCentroid = np.asarray(newSize, dtype=np.float32) / 2.0
imgStartPx = (imgCentroid - np.asarray([128, 128, 64], dtype=np.float32) / 2.0).astype(dtype=np.int32).tolist()
"""

