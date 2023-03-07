import torch
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int

import math
import numbers
import random
import warnings
from collections.abc import Sequence
from typing import Tuple, List, Optional

import torch
from torch import Tensor

class Compose:
    """Composes several transforms together. This transform does not support torchscript.
    Please, see the note below.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.PILToTensor(),
        >>>     transforms.ConvertImageDtype(torch.float),
        >>> ])

    .. note::
        In order to script the transformations, please use ``torch.nn.Sequential`` as below.

        >>> transforms = torch.nn.Sequential(
        >>>     transforms.CenterCrop(10),
        >>>     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>> )
        >>> scripted_transforms = torch.jit.script(transforms)

        Make sure to use only scriptable transformations, i.e. that work with ``torch.Tensor``, does not require
        `lambda` functions or ``PIL.Image``.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img1, img2, mask1, mask2):
        for t in self.transforms:
            img1, img2, mask1, mask2 = t(img1, img2, mask1, mask2)
        return img1, img2, mask1, mask2

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class RandomVerticalFlip(torch.nn.Module):
    """Vertically flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img1, img2, mask1, mask2):
        """
        Args:
            img, mask (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.vflip(img1), F.vflip(img2), F.vflip(mask1), F.vflip(mask2)
        return img1, img2, mask1, mask2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomHorizontalFlip(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img1, img2, mask1, mask2):
        """
        Args:
            img, mask (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            return F.hflip(img1), F.hflip(img2), F.hflip(mask1), F.hflip(mask2)
        return img1, img2, mask1, mask2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))


def _setup_angle(x, name, req_sizes=(2, )):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]

class RandomRotation(torch.nn.Module):
    """Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number): Pixel fill value for the area outside the rotated
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
        resample (int, optional): deprecated argument and will be removed since v0.10.0.
            Please use the ``interpolation`` parameter instead.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(
        self, degrees, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None
    ):
        super().__init__()
        if resample is not None:
            warnings.warn(
                "Argument resample is deprecated and will be removed since v0.10.0. Please, use interpolation instead"
            )
            interpolation = _interpolation_modes_from_int(resample)

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2, ))

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2, ))

        self.center = center

        self.resample = self.interpolation = interpolation
        self.expand = expand

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    @staticmethod
    def get_params(degrees: List[float]) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle

    def forward(self, img1, img2, mask1, mask2):
        """
        Args:
            img, mask (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        fill1 = self.fill
        fill2 = self.fill
        if isinstance(img1, Tensor) and isinstance(img2, Tensor) and isinstance(mask1, Tensor) and isinstance(mask2, Tensor):
            if isinstance(fill1, (int, float)):
                fill1 = [float(fill1)] * F.get_image_num_channels(img1)
                fill2 = [float(fill2)] * F.get_image_num_channels(img2)
            else:
                fill1 = [float(f) for f in fill1]
                fill2 = [float(f) for f in fill2]
        angle = self.get_params(self.degrees)

        return F.rotate(img1, angle, self.resample, self.expand, self.center, fill1), F.rotate(img2, angle, self.resample, self.expand, self.center, fill2), F.rotate(mask1, angle, self.resample, self.expand, self.center, fill1), F.rotate(mask2, angle, self.resample, self.expand, self.center, fill2)

    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', interpolation={0}'.format(interpolate_str)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        if self.fill is not None:
            format_string += ', fill={0}'.format(self.fill)
        format_string += ')'
        return format_string

class ToTensor1:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This transform does not support torchscript.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.

    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks. See the `references`_ for implementing the transforms for image masks.

    .. _references: https://github.com/pytorch/vision/tree/main/references/segmentation
    """

    def __call__(self, img1, img2 , mask1, mask2):
        """
        Args:
            img, mask (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(img1), F.to_tensor(img2), F.to_tensor(mask1), F.to_tensor(mask2)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomBrightness(torch.nn.Module):
    """How much to adjust the brightness. Can be
            any non-negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Args:
        p (float): probability of the image being increased brightness. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img1, img2, mask1, mask2):
        """
        Args:
            img, mask (PIL Image or Tensor): Image to be changed brightness.

        Returns:
            PIL Image or Tensor: Random increase in brightness between 1 and 2.
        """
        if torch.rand(1) < self.p:
            bright_factor = float(torch.rand(1)) + 1
            return F.adjust_brightness(img1,bright_factor), F.adjust_brightness(img2,bright_factor), mask1, mask2
        return img1, img2, mask1, mask2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomContrast(torch.nn.Module):
    """How much to adjust the contrast. Can be any
            non-negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Args:
        p (float): probability of the image being increased contrast. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img1, img2, mask1, mask2):
        """
        Args:
            img, mask (PIL Image or Tensor): Image to be changed contrast.

        Returns:
            PIL Image or Tensor: Random increase in contrast between 1 and 2.
        """
        if torch.rand(1) < self.p:
            factor = float(torch.rand(1)) + 1
            return F.adjust_contrast(img1,factor), F.adjust_contrast(img2,factor), mask1, mask2
        return img1, img2, mask1, mask2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomSaturation(torch.nn.Module):
    """How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Args:
        p (float): probability of the image being increased saturation. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img1, img2, mask1, mask2):
        """
        Args:
            img, mask (PIL Image or Tensor): Image to be changed saturation.

        Returns:
            PIL Image or Tensor: Random increase in saturation between 1 and 2.
        """
        if torch.rand(1) < self.p:
            factor = float(torch.rand(1)) + 1
            return F.adjust_saturation(img1,factor), F.adjust_saturation(img2,factor), mask1, mask2
        return img1, img2, mask1, mask2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomHue(torch.nn.Module):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Args:
        p (float): probability of the image being increased hue. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img1, img2, mask1, mask2):
        """
        Args:
            img, mask (PIL Image or Tensor): Image to be changed hue.

        Returns:
            PIL Image or Tensor: Random increase in hue between 0 and 0.1 .
        """
        if torch.rand(1) < self.p:
            factor = float(torch.rand(1))/10
            return F.adjust_hue(img1,factor), F.adjust_hue(img2,factor), mask1, mask2
        return img1, img2, mask1, mask2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomGamma(torch.nn.Module):
    """
    ntensities in RGB mode are adjusted
    based on the following equation:

    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}

    gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
            gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter.
    Args:
        p (float): probability of the image being adjusted gamma. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img1, img2, mask1, mask2):
        """
        Args:
            img, mask (PIL Image or Tensor): Image to be changed gamma.

        Returns:
            PIL Image or Tensor: Random adjust in gamma between 0.1 and 2.
        """
        if torch.rand(1) < self.p:
            factor = float(torch.rand(1)) + 0.1 + float(torch.rand(1))
            return F.adjust_gamma(img1,factor,1), F.adjust_gamma(img2,factor,1), mask1, mask2
        return img1, img2, mask1, mask2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomGaussianBlur(torch.nn.Module):
    """Performs Gaussian blurring on the image by given kernel.
    Args:
        p (float): probability of the image being increased gaussian. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img1, img2, mask1, mask2):
        """
        Args:
            img, mask (PIL Image or Tensor): Image to be changed using gaussian noise.

        Returns:
            PIL Image or Tensor: Random adjustment by gaussian blur.
        """
        if torch.rand(1) < self.p:
            factor = float(torch.rand(1)) + 1
            return F.gaussian_blur(img1,[3,3],[factor,factor]), F.gaussian_blur(img2,[3,3],[factor,factor]), mask1, mask2
        return img1, img2, mask1, mask2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomSharpness(torch.nn.Module):
    """How much to adjust the sharpness. Can be
            any non-negative number. 0 gives a blurred image, 1 gives the
            original image while 2 increases the sharpness by a factor of 2.
    Args:
        p (float): probability of the image being adjusted sharpness. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img1, img2, mask1, mask2):
        """
        Args:
            img, mask (PIL Image or Tensor): Image to be changed sharpness.

        Returns:
            PIL Image or Tensor: Random adjustment by sharpness.
        """
        if torch.rand(1) < self.p:
            factor = float(torch.rand(1)) + 1
            return F.adjust_sharpness(img1,factor), F.adjust_sharpness(img2,factor), mask1, mask2
        return img1, img2, mask1, mask2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)