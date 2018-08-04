import os

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from sample.read import readImage

basePath = os.path.abspath('..')
trainPath = os.path.join(basePath, 'data/train')

def showImage(npImg):
    n = npImg.shape[0]
    fig = plt.figure(figsize=(30, 30))
    cols = 8
    rows = n / cols + 1
    for i in range(1, n + 1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(npImg[i - 1], cmap='gray')
    plt.show()

def saveImage(npImg):
    n = npImg.shape[0]
    fig = plt.figure(figsize=(30, 30))
    cols = 8
    rows = n / cols + 1
    for i in range(1, n + 1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(npImg[i - 1], cmap='gray')
    plt.savefig('fig.png')

sitkImg = readImage(os.path.join(trainPath, 'Case00.mhd'))
sitkGT = readImage(os.path.join(trainPath, 'Case00_segmentation.mhd'))
print('sitkImg: {}'.format(sitkImg.GetSize()))
print('sitkGT: {}'.format(sitkGT.GetSize()))

npImg = sitk.GetArrayFromImage(sitkImg)
npGT = sitk.GetArrayFromImage(sitkGT)
saveImage(npImg)
saveImage(npGT)
