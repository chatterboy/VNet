import os
import numpy as np
import SimpleITK as sitk

class Batch:
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.index = 0

    def _reach_to_last(self):
        return self.x.shape[0] < self.index + self.batch_size

    def _shuffle(self):
        print("Shuffling dataset")
        new_index = [idx for idx in range(self.x.shape[0])]
        np.random.shuffle(new_index)
        print(new_index)
        new_x = np.zeros(self.x.shape)
        new_y = np.zeros(self.y.shape)
        for i in range(self.x.shape[0]):
            new_x[i, :, :, :, :] = self.x[new_index[i], :, :, :, :]
            new_y[i, :, :, :, :] = self.y[new_index[i], :, :, :, :]
        self.x = new_x
        self.y = new_y

    def next_to(self):
        if self._reach_to_last():
            self.index = 0
            self._shuffle()
        batch_x = self.x[self.index:self.index + self.batch_size]
        batch_y = self.y[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return batch_x, batch_y

class DataManager:
    def __init__(self, paths):
        """
            There are some paths as well as for train data
        :param paths: dictionary,
        """
        self._paths = paths

    def _read_image_files(self, path):
        """
        :param path: string, the path of the directory includes image files
        :return self.image_files: list of string, a list of image files (names)
        """
        self._image_files = [f for f in os.listdir(path) if 'raw' not in f and 'segmentation' not in f]
        return self._image_files

    def _read_label_files(self, path):
        """
        :param path: string, the path of the directory includes image files
        :return self.label_files: list of string, a list of label files (names)
        """
        self._label_files = []
        for name, ext in [os.path.splitext(f) for f in self._read_image_files(path)]:
            self._label_files.append(''.join([name, '_segmentation', ext]))
        return self._label_files

    def _read_sitk_image(self, path, dtype=sitk.sitkFloat32):
        """
        :param path:
        :param dtype:
        :return:
        """
        return sitk.Cast(sitk.ReadImage(path), dtype)

    def _read_sitk_images(self, path, files, dtype=sitk.sitkFloat32):
        """
        :param path: string, the path of the driectory to read the image files
        :param files: list of string, a list of image files (names)
        :param dtype:
        :return self.sitk_images: list of sitk_images
        """
        self._sitk_images = [self._read_sitk_image(os.path.join(path, f), dtype) for f in files]
        return self._sitk_images

    def _read_sitk_labels(self, path, files, dtype=sitk.sitkFloat32):
        """
        :param path:
        :param files:
        :param dtype:
        :return:
        """
        self._sitk_labels = [self._read_sitk_image(os.path.join(path, f), dtype) for f in files]
        return self._sitk_labels

    def load_train_data(self):
        """
            Load train data from the path specified, the data consists of images and ground-truths to train a model
        :return:
        """
        train_data_path = self._paths['train_data_path']
        image_files = self._read_image_files(train_data_path)
        label_files = self._read_label_files(train_data_path)
        self._read_sitk_images(train_data_path, image_files)
        self._read_sitk_labels(train_data_path, label_files)

    def get_numpy_from_sitk_image(self, sitk_image):
        """
        :param sitk_image: a sitk_image
        :return: a numpy_image
        """
        return sitk.GetArrayFromImage(sitk_image)

    def get_numpy_from_sitk_images(self, sitk_images):
        """
        :param sitk_images: a list of sitk_images
        :return: a list of numpy_images
        """
        return [self.get_numpy_from_sitk_image(sitk_image) for sitk_image in sitk_images]

    def get_sitk_images(self):
        return self._sitk_images

    def get_sitk_labels(self):
        return self._sitk_labels

    def get_numpy_images(self):
        return [sitk.GetArrayFromImage(img) for img in self._sitk_images]

    def get_numpy_labels(self):
        return [sitk.GetArrayFromImage(gt) for gt in self._sitk_labels]