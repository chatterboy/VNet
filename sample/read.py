import os
import SimpleITK as sitk

def readImage(path, dtype=sitk.sitkFloat32):
    return sitk.Cast(sitk.ReadImage(path), dtype)

basePath = os.path.abspath('..')
trainPath = os.path.join(basePath, 'data/train')

sitkImg = readImage(os.path.join(trainPath, 'Case00.mhd'))
npImg = sitk.GetArrayFromImage(sitkImg)

print(npImg.shape)