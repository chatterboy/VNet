import os
import numpy as np
import SimpleITK as sitk

"""
    0 1 2 3 4
    ->
    3 0 4 1 2

    1. [0 1] [2 3]
    2. [3 0] [4 1]
"""

class Batch:
    def __init__(self, x, y, batch_size=1):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.index = 0

    def reachToLast(self):
        return self.x.shape[0] < self.index + self.batch_size

    def nextTo(self):
        if self.reachToLast():
            self.index = 0
            np.random.shuffle(self.x)
            np.random.shuffle(self.y)
        batch_x = self.x[self.index : self.index + self.batch_size]
        batch_y = self.y[self.index : self.index + self.batch_size]
        return batch_x, batch_y

class DataManager:
    """

    """
    def __init__(self, config):
        """

        """
        self.config = config

    def get_file_list(self):
        """

        :return:
        """
        path = os.path.join(self.config['base_path'], self.config['train_path'])
        self.file_list = \
            [f for f in os.listdir(path) if 'raw' not in f and 'segmentation' not in f]
        print(self.file_list)

    def get_gt_list(self):
        """

        :return:
        """
        self.gt_list = []
        for f in self.file_list:
            name, ext = os.path.splitext(f)
            self.gt_list.append(name + '_segmentation' + ext)
        print(self.gt_list)

    def load_images(self):
        """

        :return:
        """
        self.file_dict = dict()
        for f in self.file_list:
            self.file_dict[f] = \
                sitk.Cast(sitk.ReadImage(os.path.join(self.config['base_path'],
                                                      self.config['train_path'], f)),
                          sitk.sitkFloat32)

    def load_labels(self):
        """

        :return:
        """
        self.gt_dict = dict()
        for f in self.gt_list:
            self.gt_dict[f] = \
                sitk.Cast(sitk.ReadImage(os.path.join(self.config['base_path'],
                                                      self.config['train_path'], f)),
                          sitk.sitkFloat32)

    def load_train(self):
        """

        :return:
        """
        self.get_file_list()
        self.get_gt_list()
        self.load_images()
        self.load_labels()

    def getNumpyData(self, dat):
        ret = dict()
        for key in dat:
            ret[key] = np.zeros([128, 128, 64], dtype=np.float32)

            img = dat[key]

            rescalFilt = sitk.RescaleIntensityImageFilter()
            rescalFilt.SetOutputMaximum(1)
            rescalFilt.SetOutputMinimum(0)

            imgRescaled = rescalFilt.Execute(img)

            # we rotate the image according to its transformation using the direction and according to the final spacing we want
            factor = np.asarray(img.GetSpacing()) / [1.0, 1.0, 1.5]

            factorSize = np.asarray(img.GetSize() * factor, dtype=float)

            newSize = np.max([factorSize, [128, 128, 64]], axis=0).astype(dtype=np.uint32).tolist()

            T = sitk.AffineTransform(3)
            T.SetMatrix(img.GetDirection())

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(imgRescaled)
            resampler.SetOutputSpacing([1.0, 1.0, 1.5])
            resampler.SetSize(newSize)
            resampler.SetInterpolator(sitk.sitkLinear)

            imgResampled = resampler.Execute(imgRescaled)

            imgCentroid = np.asarray(newSize, dtype=np.float32) / 2.0

            imgStartPx = (imgCentroid - np.asarray([128, 128, 64], dtype=np.float32) / 2.0).astype(dtype=np.int32).tolist()

            regionExtractor = sitk.RegionOfInterestImageFilter()
            regionExtractor.SetSize([128, 128, 64])
            regionExtractor.SetIndex(imgStartPx)

            imgResampledCropped = regionExtractor.Execute(imgResampled)

#            ret[key] = np.transpose(sitk.GetArrayFromImage(imgResampledCropped).astype(dtype=np.float32), [2, 1, 0])
            ret[key] = sitk.GetArrayFromImage(imgResampledCropped).astype(dtype=np.float32)

        return ret


    def get_numpy_images(self):
        """

        :return:
        """
        dat = self.getNumpyData(self.file_dict)
#        ret = np.fromiter(dat.values(), dtype=dtype)
        ret = [dat[key] for key in dat]
        return ret


    def get_numpy_labels(self):
        """

        :return:
        """
        dat = self.getNumpyData(self.gt_dict)
#        ret = np.fromiter(dat.values(), dtype=np.float32)
        ret = [dat[key] for key in dat]
        return ret