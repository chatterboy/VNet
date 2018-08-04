import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from src.utility.dm import DataManager

"""
def save(self, logits, batch_x, batch_y, config, epoch):
    def _save(name, epoch, config, npImg):
        _, depth, height, width, _ = npImg.shape
        for i in range(2):
            fig = plt.figure(figsize=(30, 30))
            cols = 8
            rows = depth / cols + 1
            for j in range(1, depth + 1):
                fig.add_subplot(rows, cols, j)
                if name == 'gt':
                    plt.imshow(npImg[0, j - 1, :, :, i], cmap='gray', vmin=0, vmax=1)
                else:
                    plt.imshow(npImg[0, j - 1, :, :, i], cmap='gray')
            plt.savefig(os.path.join(config['base_path'], 'result',
                                     ''.join([s for s in [str(epoch), '-', ['bg', 'fg'][i], '-', name, '.png']])))

    def _save_tr(name, epoch, config, npImg):
        _, depth, height, width, _ = npImg.shape
        fig = plt.figure(figsize=(30, 30))
        cols = 8
        rows = depth / cols + 1
        for i in range(1, depth + 1):
            fig.add_subplot(rows, cols, i)
            plt.imshow(npImg[0, i - 1, :, :, 0], cmap='gray', vmin=0, vmax=1)
        plt.savefig(os.path.join(config['base_path'], 'result',
                                 ''.join([s for s in [str(epoch), '-', name, '.png']])))

    _save_tr('tr', epoch, config, batch_x)
    _save('img', epoch, config, logits)
    _save('gt', epoch, config, batch_y)
"""

def save_np_images(name, path, np_images):
    def _save_np_img(name, index, path, np_img):
        depth, height, width = np_img.shape
        fig = plt.figure(figsize=(30, 30))
        cols = 8
        rows = depth / cols + 1
        for i in range(1, depth + 1):
            fig.add_subplot(rows, cols, i)
            plt.imshow(np_img[i - 1, :, :], cmap='gray', vmin=0, vmax=1)
        plt.savefig(os.path.join(path, ''.join([s for s in [str(index), '-', name, '.png']])))
        plt.close()
    index = 1
    for np_img in np_images:
        _save_np_img(name, index, path, np_img)
        index = index + 1

def change_sitk_to_np(sitk_images):
    return [sitk.GetArrayFromImage(v) for v in sitk_images]

def rescale_sitk_images(sitk_images):
    def _rescale_sitk_image(sitk_image):
        rescaler = sitk.RescaleIntensityImageFilter()
        rescaler.SetOutputMinimum(0)
        rescaler.SetOutputMaximum(1)
        return rescaler.Execute(sitk_image)
    return [_rescale_sitk_image(v) for v in sitk_images]

basePath = os.path.abspath('..')
trainPath = os.path.join(basePath, 'data/train')

config = dict()
config['base_path'] = os.path.abspath('..')
config['train_path'] = os.path.join(basePath, 'data/train')

result_path = os.path.join(config['base_path'], 'result')

dm = DataManager(config)

dm.load_train()

sitk_images = dm.get_sitk_images()
sitk_gts = dm.get_sitk_labels()
np_images = dm.get_numpy_images()
np_gts = dm.get_numpy_labels()

save_np_images('raw_img', result_path, change_sitk_to_np(rescale_sitk_images(sitk_images)))
save_np_images('raw_gt', result_path, change_sitk_to_np(rescale_sitk_images(sitk_gts)))
save_np_images('img', result_path, np_images)
save_np_images('gt', result_path, np_gts)