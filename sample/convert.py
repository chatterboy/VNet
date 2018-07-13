import os
import SimpleITK as sitk

from sample.read import readImage

def convertImage(sitkImg):
    

basePath = os.path.abspath('..')
trainPath = os.path.join(basePath, 'data/train')

sitkImg = convertImage(readImage(os.path.join(trainPath, 'Case00.mhd')))

npImg = sitk.GetArrayFromImage(sitkImg)