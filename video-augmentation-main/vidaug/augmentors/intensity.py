"""
Augmenters that apply transformations on the pixel intensities.

To use the augmenters, clone the complete repo and use
`from vidaug import augmenters as va`
and then e.g. :
    seq = va.Sequential([ va.RandomRotate(30),
                          va.RandomResize(0.2)  ])

List of augmenters:
    * InvertColor
    * Add
    * Multiply
    * Pepper
    * Salt
    * Brightness
    * Contrast
    * Gamma
    * HueSaturation
    * GaussianNoise
"""


import numpy as np
import random
import PIL
from PIL import ImageOps
import cv2



class InvertColor(object):
    """
    Inverts the color of the video.
    """

    def __call__(self, clip):
        if isinstance(clip[0], np.ndarray):
            return [np.invert(img) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            inverted = [ImageOps.invert(img) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))

        return inverted


class Add(object):
    """
    Add a value to all pixel intesities in an video.

    Args:
        value (int): The value to be added to pixel intesities.
    """

    def __init__(self, value=0):
        if value > 255 or value < -255:
            raise TypeError('The video is blacked or whitened out since ' +
                            'value > 255 or value < -255.')
        self.value = value

    def __call__(self, clip):

        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            image = clip[i].astype(np.int32)
            image += self.value
            image = np.where(image > 255, 255, image)
            image = np.where(image < 0, 0, image)
            image = image.astype(np.uint8)
            data_final.append(image.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class Multiply(object):
    """
    Multiply all pixel intensities with given value.
    This augmenter can be used to make images lighter or darker.

    Args:
        value (float): The value with which to multiply the pixel intensities
        of video.
    """

    def __init__(self, value=1.0):
        if value < 0.0:
            raise TypeError('The video is blacked out since for value < 0.0')
        self.value = value

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            image = clip[i].astype(np.float64)
            image *= self.value
            image = np.where(image > 255, 255, image)
            image = np.where(image < 0, 0, image)
            image = image.astype(np.uint8)
            data_final.append(image.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class Pepper(object):
    """
    Augmenter that sets a certain fraction of pixel intensities to 0, hence
    they become black.

    Args:
        ratio (int): Determines number of black pixels on each frame of video.
        Smaller the ratio, higher the number of black pixels.
    """
    def __init__(self, ratio=100):
        self.ratio = ratio

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            img = clip[i].astype(np.float)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 0, img)
            data_final.append(img.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final

class Salt(object):
    """
    Augmenter that sets a certain fraction of pixel intesities to 255, hence
    they become white.

    Args:
        ratio (int): Determines number of white pixels on each frame of video.
        Smaller the ratio, higher the number of white pixels.
   """
    def __init__(self, ratio=100):
        self.ratio = ratio

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        data_final = []
        for i in range(len(clip)):
            img = clip[i].astype(np.float)
            img_shape = img.shape
            noise = np.random.randint(self.ratio, size=img_shape)
            img = np.where(noise == 0, 255, img)
            data_final.append(img.astype(np.uint8))

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class Brightness(object):
    """
    Adjusts the brightness of the video frames.
    Args:
        factor (float): Factor to adjust brightness. >1 increases, <1 decreases.
    """
    def __init__(self, factor=1.0):
        self.factor = factor
    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            return [PIL.ImageEnhance.Brightness(img).enhance(self.factor) for img in clip]
        else:
            data_final = []
            for img in clip:
                img = img.astype(np.float32) * self.factor
                img = np.clip(img, 0, 255).astype(np.uint8)
                data_final.append(img)
            return data_final

class Contrast(object):
    """
    Adjusts the contrast of the video frames.
    Args:
        factor (float): Factor to adjust contrast. >1 increases, <1 decreases.
    """
    def __init__(self, factor=1.0):
        self.factor = factor
    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            return [PIL.ImageEnhance.Contrast(img).enhance(self.factor) for img in clip]
        else:
            data_final = []
            for img in clip:
                mean = np.mean(img, axis=(0,1), keepdims=True)
                img = (img - mean) * self.factor + mean
                img = np.clip(img, 0, 255).astype(np.uint8)
                data_final.append(img)
            return data_final

class Gamma(object):
    """
    Adjusts the gamma of the video frames.
    Args:
        gamma (float): Gamma correction factor.
    """
    def __init__(self, gamma=1.0):
        self.gamma = gamma
    def __call__(self, clip):
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        data_final = []
        for img in clip:
            if is_PIL:
                img = np.array(img.convert('RGB'))
            # Ensure image is uint8 and 3 channels
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = img[..., :3]
            img = cv2.LUT(img, table)
            if is_PIL:
                img = PIL.Image.fromarray(img)
            data_final.append(img)
        return data_final

class HueSaturation(object):
    """
    Adjusts the hue and saturation of the video frames.
    Args:
        hue_shift (float): Amount to shift hue (-0.5 to 0.5).
        sat_mult (float): Saturation multiplier.
    """
    def __init__(self, hue_shift=0.0, sat_mult=1.0):
        self.hue_shift = hue_shift
        self.sat_mult = sat_mult
    def __call__(self, clip):
        data_final = []
        for img in clip:
            is_PIL = isinstance(img, PIL.Image.Image)
            if is_PIL:
                img = np.array(img.convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img = img.astype(np.float32)
            img[..., 0] = (img[..., 0] + self.hue_shift * 180) % 180
            img[..., 1] = np.clip(img[..., 1] * self.sat_mult, 0, 255)
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            if is_PIL:
                img = PIL.Image.fromarray(img)
            data_final.append(img)
        return data_final

class GaussianNoise(object):
    """
    Adds Gaussian noise to video frames.
    Args:
        sigma (float): Standard deviation of the noise.
    """
    def __init__(self, sigma=10.0):
        self.sigma = sigma
    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]
        data_final = []
        for img in clip:
            noise = np.random.normal(0, self.sigma, img.shape)
            img = img.astype(np.float32) + noise
            img = np.clip(img, 0, 255).astype(np.uint8)
            data_final.append(img)
        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final
