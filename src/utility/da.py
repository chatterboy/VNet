import numpy as np
import SimpleITK as sitk

class DataAugmentation:
    def apply_hist_equalization_to_sitk_image(self, sitk_image):
        """

        :param sitk_image: a 3D sitk_image
        :return equalizaed_numpy_image: a 3D numpy_image
        """
        return self.apply_hist_equalization_to_numpy_image(sitk.GetArrayFromImage(sitk_image))

    def apply_hist_equalization_to_numpy_image(self, numpy_image):
        """

        :param numpy_image: a 3-D numpy_image
        :return eqaulized_numpy_image: a 3-D numpy_image
        """
        flattened_numpy_image = numpy_image.flatten()
        freq = [0] * 256
        for x in flattened_numpy_image:
            freq[int(x * 255)] += 1
        accum_sum = [0] * 256
        accum_sum[0] = freq[0]
        for i in range(1, accum_sum.__len__()):
            accum_sum[i] = accum_sum[i - 1] + freq[i]
        for i in range(accum_sum.__len__()):
            accum_sum[i] /= flattened_numpy_image.__len__()
        eqaulized_numpy_image = np.zeros(numpy_image.shape, dtype=np.float32)
        depth, height, width = numpy_image.shape
        for d in range(depth):
            for h in range(height):
                for w in range(width):
                    eqaulized_numpy_image[d, h, w] = accum_sum[int(numpy_image[d, h, w] * 255)]
        return eqaulized_numpy_image

    def apply_hist_equalization_to_sitk_images(self, sitk_images):
        """
        :param sitk_images: a list of 3-D sitk_images
        :return numpy_images: a list of 3-D numpy_images
        """
        return [self.apply_hist_equalization_to_sitk_image(sitk_image) for sitk_image in sitk_images]

    def apply_hist_eqaulization_to_numpy_images(self, numpy_images):
        """
        :param numpy_images: a list of 3-D numpy_images
        :return a list of 3-D numpy_images: a list of 3-D numpy_images that each is applied by standard histogram eqaulization
        """
        return [self.apply_hist_equalization_to_numpy_image(numpy_image) for numpy_image in numpy_images]

    def rescale_sitk_image(self, sitk_image, minv=0, maxv=1):
        """
        :param sitk_image:
        :param minv:
        :param maxv:
        :return: a rescaled sitk_image to a specific range
        """
        rescaler = sitk.RescaleIntensityImageFilter()
        rescaler.SetOutputMinimum(minv)
        rescaler.SetOutputMaximum(maxv)
        return rescaler.Execute(sitk_image)

    def rescale_sitk_images(self, sitk_images, minv=0, maxv=1):
        """

        :param sitk_images: list of sitk_images
        :param minv:
        :param maxv:
        :return: list of sitk_images done rescaling
        """
        return [self.rescale_sitk_image(img, minv, maxv) for img in sitk_images]

    def resample_sitk_image_to_desired_size(self, sitk_image, output_size):
        """
        :param sitk_image:
        :param output_size:
        :return: a resampled sitk_image that change to desired size
        """
        physical_size = [x[0] * x[1] for x in zip(sitk_image.GetSize(), sitk_image.GetSpacing())]
        output_spacing = [x[0] / x[1] for x in zip(physical_size, output_size)]
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(output_size)
        resampler.SetOutputOrigin(sitk_image.GetOrigin())
        resampler.SetOutputSpacing(output_spacing)
        resampler.SetOutputDirection(sitk_image.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        return resampler.Execute(sitk_image)

    def resample_sitk_images_to_desired_size(self, sitk_images, output_size):
        """
        :param sitk_images: list of sitk_images
        :param output_size:
        :return: list of sitk_images
        """
        return [self.resample_sitk_image_to_desired_size(img, output_size) for img in sitk_images]