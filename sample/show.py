import os
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

sitkImg = readImage(os.path.join(trainPath, 'Case00.mhd'))
npImg = sitk.GetArrayFromImage(sitkImg)

showImage(npImg)