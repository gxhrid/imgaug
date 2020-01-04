"""
Augmenters that wrap methods from ``bethgelab.imagecorruptions`` package.

See https://github.com/bethgelab/imagecorruptions for the package.

List of augmenters:

    * GaussianNoise
    * ShotNoise
    * ImpulseNoise
    * SpeckleNoise
    * GaussianBlur
    * GlassBlur
    * DefocusBlur
    * MotionBlur
    * ZoomBlur
    * Fog
    * Frost
    * Snow
    * Spatter
    * Contrast
    * Brightness
    * Saturate
    * JpegCompression
    * Pixelate
    * ElasticTransform

.. note::

    The functions provided here have identical outputs to the ones in
    ``imagecorruptions`` when called using the ``corrupt()`` function of
    that package. E.g. the outputs are always ``uint8`` and not
    ``float32`` or ``float64``.

Example usage::

    >>> import imgaug as ia
    >>> import imgaug.augmenters as iaa
    >>> import numpy as np
    >>> image = np.zeros((64, 64, 3), dtype=np.uint8)
    >>> names, funcs = iaa.imgcorrupt.get_corruption_names("validation")
    >>> for name, func in zip(names, funcs):
    >>>     image_aug = func(image, severity=5, seed=1)
    >>>     image_aug = ia.draw_text(image_aug, x=20, y=20, text=name)
    >>>     ia.imshow(image_aug)

    Use e.g. ``iaa.imgcorrupt.GaussianNoise(severity=2)(images=...)`` to
    create and apply a specific augmenter.

"""
from __future__ import print_function, division, absolute_import

import warnings

import numpy as np

from .. import dtypes as iadt
from .. import random as iarandom
from .. import parameters as iap
from . import meta

# TODO add optional dependency

_MISSING_PACKAGE_ERROR_MSG = (
    "Could not import package `imagecorruptions`. This is an optional "
    "dependency of imgaug and must be installed manually in order "
    "to use augmenters from `imgaug.augmenters.imgcorrupt`. "
    "Use e.g. `pip install imagecorruptions` to install it. See also "
    "https://github.com/bethgelab/imagecorruptions for the repository "
    "of the package."
)


def _clipped_zoom_no_scipy_warning(img, zoom_factor):
    from scipy.ndimage import zoom as scizoom

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*output shape of zoom.*")

        # clipping along the width dimension:
        ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
        top0 = (img.shape[0] - ch0) // 2

        # clipping along the height dimension:
        ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
        top1 = (img.shape[1] - ch1) // 2

        img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1],
                      (zoom_factor, zoom_factor, 1), order=1)

        return img


def _call_imgcorrupt_func(fname, seed, convert_to_pil, *args, **kwargs):
    """Apply an ``imagecorruptions`` function.

    The dtype support below is basically a placeholder to which the
    augmentation functions can point to decrease the amount of documentation.

    dtype support::

        * ``uint8``: yes; indirectly tested (1)
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

        - (1) Tested by comparison with function in ``imagecorruptions``
              package.

    """
    # import imagecorruptions, note that it is an optional dependency
    try:
        # imagecorruptions sets its own warnings filter rule via
        # warnings.simplefilter(). That rule is the in effect for the whole
        # program and not just the module. So to prevent that here
        # we use catch_warnings(), which uintuitively does not by default
        # catch warnings but saves and restores the warnings filter settings.
        with warnings.catch_warnings():
            import imagecorruptions.corruptions as corruptions
    except ImportError:
        raise ImportError(_MISSING_PACKAGE_ERROR_MSG)

    # Monkeypatch clip_zoom() as that causes warnings in some scipy versions,
    # and the implementation here suppresses these warnings. They suppress
    # all UserWarnings on a module level instead, which seems very exhaustive.
    corruptions.clipped_zoom = _clipped_zoom_no_scipy_warning

    image = args[0]

    iadt.gate_dtypes(
        image,
        allowed=["uint8"],
        disallowed=["bool",
                    "uint16", "uint32", "uint64", "uint128", "uint256",
                    "int8", "int16", "int32", "int64", "int128", "int256",
                    "float16", "float32", "float64", "float96", "float128",
                    "float256"],
        augmenter=None)

    input_shape = image.shape

    height, width = input_shape[0:2]
    assert height >= 32 and width >= 32, (
        "Expected the provided image to have a width and height of at least "
        "32 pixels, as that is the lower limit that the wrapped "
        "imagecorruptions functions use. Got shape %s." % (image.shape,))

    ndim = image.ndim
    assert ndim == 2 or (ndim == 3 and (image.shape[2] in [1, 3])), (
        "Expected input image to have shape (height, width) or "
        "(height, width, 1) or (height, width, 3). Got shape %s." % (
            image.shape,))

    if ndim == 2:
        image = image[..., np.newaxis]
    if image.shape[-1] == 1:
        image = np.tile(image, (1, 1, 3))

    if convert_to_pil:
        import PIL.Image
        image = PIL.Image.fromarray(image)

    with iarandom.temporary_numpy_seed(seed):
        image_aug = getattr(corruptions, fname)(image, *args[1:], **kwargs)

    if convert_to_pil:
        image_aug = np.asarray(image_aug)

    if ndim == 2:
        image_aug = image_aug[:, :, 0]
    elif input_shape[-1] == 1:
        image_aug = image_aug[:, :, 0:1]

    # this cast is done at the end of imagecorruptions.__init__.corrupt()
    image_aug = np.uint8(image_aug)

    return image_aug


def get_corruption_names(subset="common"):
    """Get a named subset of image corruption functions.

    .. note::

        This function returns the augmentation names (as strings) *and* the
        corresponding augmentation functions, while ``get_corruption_names()``
        in ``imagecorruptions`` only returns the augmentation names.

    Parameters
    ----------
    subset : {'common', 'validation', 'all'}, optional.
        Name of the subset of image corruption functions.

    Returns
    -------
    list of str
        Names of the corruption methods, e.g. "gaussian_noise".

    list of callable
        Function corresponding to the name. Is one of the
        ``apply_*()`` functions in this module. Apply e.g.
        via ``func(image, severity=2, seed=123)``.

    """
    # import imagecorruptions, note that it is an optional dependency
    try:
        # imagecorruptions sets its own warnings filter rule via
        # warnings.simplefilter(). That rule is the in effect for the whole
        # program and not just the module. So to prevent that here
        # we use catch_warnings(), which uintuitively does not by default
        # catch warnings but saves and restores the warnings filter settings.
        with warnings.catch_warnings():
            import imagecorruptions
    except ImportError:
        raise ImportError(_MISSING_PACKAGE_ERROR_MSG)

    cnames = imagecorruptions.get_corruption_names(subset)
    funcs = [globals()["apply_%s" % (cname,)] for cname in cnames]

    return cnames, funcs


# ----------------------------------------------------------------------------
# Corruption functions
# ----------------------------------------------------------------------------
# These functions could easily be created dynamically, especially templating
# the docstrings would save many lines of code. It is intentionally not done
# here for the same reasons as in case of the augmenters. See the comment
# further below at the start of the augmenter section for details.

def apply_gaussian_noise(x, severity=1, seed=None):
    """Apply ``gaussian_noise`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("gaussian_noise", seed, False, x, severity)


def apply_shot_noise(x, severity=1, seed=None):
    """Apply ``shot_noise`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("shot_noise", seed, False, x, severity)


def apply_impulse_noise(x, severity=1, seed=None):
    """Apply ``impulse_noise`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("impulse_noise", seed, False, x, severity)


def apply_speckle_noise(x, severity=1, seed=None):
    """Apply ``speckle_noise`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("speckle_noise", seed, False, x, severity)


def apply_gaussian_blur(x, severity=1, seed=None):
    """Apply ``gaussian_blur`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("gaussian_blur", seed, False, x, severity)


def apply_glass_blur(x, severity=1, seed=None):
    """Apply ``glass_blur`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("glass_blur", seed, False, x, severity)


def apply_defocus_blur(x, severity=1, seed=None):
    """Apply ``defocus_blur`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("defocus_blur", seed, False, x, severity)


def apply_motion_blur(x, severity=1, seed=None):
    """Apply ``motion_blur`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("motion_blur", seed, False, x, severity)


def apply_zoom_blur(x, severity=1, seed=None):
    """Apply ``zoom_blur`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("zoom_blur", seed, False, x, severity)


def apply_fog(x, severity=1, seed=None):
    """Apply ``fog`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("fog", seed, False, x, severity)


def apply_frost(x, severity=1, seed=None):
    """Apply ``frost`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("frost", seed, False, x, severity)


def apply_snow(x, severity=1, seed=None):
    """Apply ``snow`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("snow", seed, False, x, severity)


def apply_spatter(x, severity=1, seed=None):
    """Apply ``spatter`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("spatter", seed, True, x, severity)


def apply_contrast(x, severity=1, seed=None):
    """Apply ``contrast`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("contrast", seed, False, x, severity)


def apply_brightness(x, severity=1, seed=None):
    """Apply ``brightness`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("brightness", seed, False, x, severity)


def apply_saturate(x, severity=1, seed=None):
    """Apply ``saturate`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("saturate", seed, False, x, severity)


def apply_jpeg_compression(x, severity=1, seed=None):
    """Apply ``jpeg_compression`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("jpeg_compression", seed, True, x, severity)


def apply_pixelate(x, severity=1, seed=None):
    """Apply ``pixelate`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    x : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("pixelate", seed, True, x, severity)


def apply_elastic_transform(image, severity=1, seed=None):
    """Apply ``elastic_transform`` from ``imagecorruptions``.

    dtype support::

        See :func:`imgaug.augmenters.imgcorrupt._call_imgcorrupt_func`.

    Parameters
    ----------
    image : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """
    return _call_imgcorrupt_func("elastic_transform", seed, False, image,
                                 severity)


# ----------------------------------------------------------------------------
# Augmenters
# ----------------------------------------------------------------------------
# The augmenter definitions below are almost identical and mainly differ in
# the names and functions used. It would be fairly trivial to write a
# function that would create these augmenters dynamically (and one is listed
# below as a comment). The downside is that in these cases the documentation
# would also be generated dynamically, which leads to numerous problems:
# (1) users couldn't easily read the documentation while scrolling through
# the code file, (2) IDEs might not be able to use it for code suggestions,
# (3) tools like pylint can't detect and validate it, (4) the imgaug-doc
# tools to parse dtype support don't work with dynamically generated
# documentation (and neither with dynamically generated classes).
# Even though it's by far more code, it seems like the better choice overall
# to just write it out.

# Example function to dynamically generate augmenters, kept for possible
# future uses:
# def _create_augmenter(class_name, func_name):
#     func = globals()["apply_%s" % (func_name,)]
#
#     def __init__(self, severity=1, name=None, deterministic=False,
#                  random_state=None):
#         super(self.__class__, self).__init__(
#             func, severity, name=name, deterministic=deterministic,
#             random_state=random_state)
#
#     augmenter_class = type(class_name,
#                            (_ImgcorruptAugmenterBase,),
#                            {"__init__": __init__})
#
#     augmenter_class.__doc__ = """
#     Wrapper around function :func:`imagecorruption.%s`.
#
#     dtype support::
#
#         See :func:`imgaug.augmenters.imgcorrupt.apply_%s`.
#
#     Parameters
#     ----------
#     severity : int, optional
#         Strength of the corruption, with valid values being
#         ``1 <= severity <= 5``.
#
#     name : None or str, optional
#         See :func:`imgaug.augmenters.meta.Augmenter.__init__`.
#
#     deterministic : bool, optional
#         See :func:`imgaug.augmenters.meta.Augmenter.__init__`.
#
#     random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
#         See :func:`imgaug.augmenters.meta.Augmenter.__init__`.
#
#     Examples
#     --------
#     >>> import imgaug.augmenters as iaa
#     >>> aug = iaa.%s(severity=2)
#
#     Create an augmenter around :func:`imagecorruption.%s`. Apply it to
#     images using e.g. ``aug(images=[image1, image2, ...])``.
#
#     """ % (func_name, func_name, class_name, func_name)
#
#     return augmenter_class


class _ImgcorruptAugmenterBase(meta.Augmenter):
    def __init__(self, func, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(_ImgcorruptAugmenterBase, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        self.func = func
        self.severity = iap.handle_discrete_param(
            severity, "severity", value_range=(1, 5), tuple_to_uniform=True,
            list_to_choice=True, allow_floats=False)

    def _augment_batch(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        severities, seeds = self._draw_samples(len(batch.images),
                                               random_state=random_state)

        for image, severity, seed in zip(batch.images, severities, seeds):
            image[...] = self.func(image, severity=severity, seed=seed)

        return batch

    def _draw_samples(self, nb_rows, random_state):
        severities = self.severity.draw_samples((nb_rows,),
                                                random_state=random_state)
        seeds = random_state.generate_seeds_(nb_rows)

        return severities, seeds

    def get_parameters(self):
        """See :func:`imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.severity]


class GaussianNoise(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.gaussian_noise`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_gaussian_noise`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.GaussianNoise(severity=2)

    Create an augmenter around :func:`imagecorruption.gaussian_noise`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(GaussianNoise, self).__init__(
            apply_gaussian_noise, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class ShotNoise(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.shot_noise`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_shot_noise`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.ShotNoise(severity=2)

    Create an augmenter around :func:`imagecorruption.shot_noise`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(ShotNoise, self).__init__(
            apply_shot_noise, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class ImpulseNoise(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.impulse_noise`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_impulse_noise`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.ImpulseNoise(severity=2)

    Create an augmenter around :func:`imagecorruption.impulse_noise`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(ImpulseNoise, self).__init__(
            apply_impulse_noise, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class SpeckleNoise(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.speckle_noise`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_speckle_noise`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.SpeckleNoise(severity=2)

    Create an augmenter around :func:`imagecorruption.speckle_noise`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(SpeckleNoise, self).__init__(
            apply_speckle_noise, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class GaussianBlur(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.gaussian_blur`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_gaussian_blur`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.GaussianBlur(severity=2)

    Create an augmenter around :func:`imagecorruption.gaussian_blur`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(GaussianBlur, self).__init__(
            apply_gaussian_blur, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class GlassBlur(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.glass_blur`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_glass_blur`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.GlassBlur(severity=2)

    Create an augmenter around :func:`imagecorruption.glass_blur`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(GlassBlur, self).__init__(
            apply_glass_blur, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class DefocusBlur(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.defocus_blur`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_defocus_blur`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.DefocusBlur(severity=2)

    Create an augmenter around :func:`imagecorruption.defocus_blur`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(DefocusBlur, self).__init__(
            apply_defocus_blur, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class MotionBlur(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.motion_blur`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_motion_blur`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.MotionBlur(severity=2)

    Create an augmenter around :func:`imagecorruption.motion_blur`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(MotionBlur, self).__init__(
            apply_motion_blur, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class ZoomBlur(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.zoom_blur`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_zoom_blur`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.ZoomBlur(severity=2)

    Create an augmenter around :func:`imagecorruption.zoom_blur`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(ZoomBlur, self).__init__(
            apply_zoom_blur, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class Fog(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.fog`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_fog`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.Fog(severity=2)

    Create an augmenter around :func:`imagecorruption.fog`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(Fog, self).__init__(
            apply_fog, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class Frost(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.frost`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_frost`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.Frost(severity=2)

    Create an augmenter around :func:`imagecorruption.frost`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(Frost, self).__init__(
            apply_frost, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class Snow(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.snow`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_snow`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.Snow(severity=2)

    Create an augmenter around :func:`imagecorruption.snow`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(Snow, self).__init__(
            apply_snow, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class Spatter(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.spatter`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_spatter`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.Spatter(severity=2)

    Create an augmenter around :func:`imagecorruption.spatter`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(Spatter, self).__init__(
            apply_spatter, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class Contrast(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.contrast`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_contrast`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.Contrast(severity=2)

    Create an augmenter around :func:`imagecorruption.contrast`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(Contrast, self).__init__(
            apply_contrast, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class Brightness(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.brightness`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_brightness`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.Brightness(severity=2)

    Create an augmenter around :func:`imagecorruption.brightness`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(Brightness, self).__init__(
            apply_brightness, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class Saturate(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.saturate`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_saturate`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.Saturate(severity=2)

    Create an augmenter around :func:`imagecorruption.saturate`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(Saturate, self).__init__(
            apply_saturate, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class JpegCompression(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.jpeg_compression`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_jpeg_compression`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.JpegCompression(severity=2)

    Create an augmenter around :func:`imagecorruption.jpeg_compression`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(JpegCompression, self).__init__(
            apply_jpeg_compression, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class Pixelate(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.pixelate`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_pixelate`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.Pixelate(severity=2)

    Create an augmenter around :func:`imagecorruption.pixelate`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(Pixelate, self).__init__(
            apply_pixelate, severity,
            name=name, deterministic=deterministic, random_state=random_state)


class ElasticTransform(_ImgcorruptAugmenterBase):
    """
    Wrapper around function :func:`imagecorruption.elastic_transform`.

    .. note ::

        This augmenter only affects images. Other data is not changed.

    dtype support::

        See
        :func:`imgaug.augmenters.imgcorrupt.apply_elastic_transform`.

    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.imgcorrupt.ElasticTransform(severity=2)

    Create an augmenter around :func:`imagecorruption.elastic_transform`.
    Apply it to images using e.g. ``aug(images=[image1, image2, ...])``.

    """

    def __init__(self, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(ElasticTransform, self).__init__(
            apply_elastic_transform, severity,
            name=name, deterministic=deterministic, random_state=random_state)
