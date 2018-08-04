import os
import numpy as np
import SimpleITK as sitk

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
        def _get_numpy_data(sitk_data):
            rescaler = sitk.RescaleIntensityImageFilter()

            rescaler.SetOutputMinimum(0)
            rescaler.SetOutputMaximum(1)

            rescaled_gt = rescaler.Execute(sitk_data)

            new_size = [max(y[0], y[1]) for y in zip([int(x[0] * x[1] / x[2]) for x in zip(sitk_data.GetSize(), sitk_data.GetSpacing(), [1.0, 1.0, 1.5])], [128, 128, 64])]

            T = sitk.AffineTransform(3)
            T.SetMatrix(sitk_data.GetDirection())

            resampler = sitk.ResampleImageFilter()

            """
                There are two behaviors that make me confuse
                First, when i code like resampler.SetReferenceImage(...) to resampler.SetSize(...) then this work well
                Second, but when i code like resampler.SetSize(...) to resampler.SetReferenceImage(...) then this is occurred an error:
                        RuntimeError: ... Requested region is (at least partially) outside the largest possible region
                Anyway, i think the error is in the settings for resampler 
            """
            resampler.SetReferenceImage(rescaled_gt)
            resampler.SetSize(new_size)
            resampler.SetOutputSpacing([1.0, 1.0, 1.5])
            resampler.SetInterpolator(sitk.sitkLinear)

            resampled_gt = resampler.Execute(rescaled_gt)

            centroid = [x[0] / x[1] for x in zip(new_size, 3 * [2.0])]

            start_px = [int(x[0] - x[1] / x[2]) for x in zip(centroid, [128, 128, 64], 3 * [2.0])]

            region_extractor = sitk.RegionOfInterestImageFilter()

            region_extractor.SetSize([128, 128, 64])
            region_extractor.SetIndex(start_px)

            cropped_gt = region_extractor.Execute(resampled_gt)

            return sitk.GetArrayFromImage(cropped_gt)

        ret = dict()
        for key in dat:
            ret[key] = _get_numpy_data(dat[key])
        return ret


    def get_sitk_images(self):
        return [v for v in self.file_dict.values()]

    def get_sitk_labels(self):
        return [v for v in self.gt_dict.values()]

    def get_numpy_images(self):
        data = self.getNumpyData(self.file_dict)
        return np.array([v for v in data.values()], dtype=np.float32)


    def get_numpy_labels(self):
        data = self.getNumpyData(self.gt_dict)
        return np.array([v for v in data.values()], dtype=np.float32)