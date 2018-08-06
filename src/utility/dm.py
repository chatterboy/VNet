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

    def read_image_files(self, path):
        """
        :param path: string, the path of the directory includes image files
        :return self.image_files: list of string, a list of image files (names)
        """
        self.image_files = [f for f in os.listdir(path) if 'raw' not in f and 'segmentation' not in f]
        return self.image_files

    def read_label_files(self, path):
        """
        :param path: string, the path of the directory includes image files
        :return self.label_files: list of string, a list of label files (names)
        """
        self.label_files = []
        for name, ext in [os.path.splitext(f) for f in self.read_image_files(path)]:
            self.label_files.append(''.join([name, '_segmentation', ext]))
        return self.label_files

    def read_sitk_image(self, path, dtype=sitk.sitkFloat32):
        """
        :param path:
        :param dtype:
        :return:
        """
        return sitk.Cast(sitk.ReadImage(path), dtype)

    def read_sitk_images(self, path, files, dtype=sitk.sitkFloat32):
        """
        :param path: string, the path of the driectory to read the image files
        :param files: list of string, a list of image files (names)
        :param dtype:
        :return self.sitk_images: list of sitk_images
        """
        self.sitk_images = [self.read_sitk_image(os.path.join(path, f), dtype) for f in files]
        return self.sitk_images

    def read_sitk_labels(self, path, files, dtype=sitk.sitkFloat32):
        """
        :param path:
        :param files:
        :param dtype:
        :return:
        """
        self.sitk_labels = [self.read_sitk_image(os.path.join(path, f), dtype) for f in files]
        return self.sitk_labels

    def load_train_data(self):
        """
            Load train data from the path specified, the data consists of images and ground-truth to train a model
        :return:
        """
        train_data_path = os.path.join(self.config['base_path'], self.config['train_path'])
        image_files = self.read_image_files(train_data_path)
        label_files = self.read_label_files(train_data_path)
        return self.read_sitk_images(train_data_path, image_files), self.read_sitk_labels(train_data_path, label_files)

    def get_sitk_images(self):
        return self.sitk_images

    def get_sitk_labels(self):
        return self.sitk_labels

    def get_numpy_images(self):
        return [sitk.GetArrayFromImage(img) for img in self.sitk_images]

    def get_numpy_labels(self):
        return [sitk.GetArrayFromImage(gt) for gt in self.sitk_labels]