from __future__ import print_function, division, absolute_import

import warnings
import sys
# unittest only added in 3.4 self.subTest()
if sys.version_info[0] < 3 or sys.version_info[1] < 4:
    import unittest2 as unittest
else:
    import unittest
# unittest.mock is not available in 2.7 (though unittest2 might contain it?)
try:
    import unittest.mock as mock
except ImportError:
    import mock

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import six.moves as sm

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug.augmenters import blend
from imgaug.testutils import (
    keypoints_equal, reseed, assert_cbaois_equal, shift_cbaoi,
    runtest_pickleable_uint8_img)
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class Test_blend_alpha(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_alpha_is_1(self):
        img_fg = np.full((3, 3, 1), 0, dtype=np.uint8)
        img_bg = np.full((3, 3, 1), 255, dtype=np.uint8)
        img_blend = blend.blend_alpha(img_fg, img_bg, 1.0, eps=0)
        assert img_blend.dtype.name == "uint8"
        assert img_blend.shape == (3, 3, 1)
        assert np.all(img_blend == 0)

    def test_alpha_is_1_2d_arrays(self):
        img_fg = np.full((3, 3), 0, dtype=np.uint8)
        img_bg = np.full((3, 3), 255, dtype=np.uint8)
        img_blend = blend.blend_alpha(img_fg, img_bg, 1.0, eps=0)
        assert img_blend.dtype.name == "uint8"
        assert img_blend.shape == (3, 3)
        assert np.all(img_blend == 0)

    def test_alpha_is_0(self):
        img_fg = np.full((3, 3, 1), 0, dtype=np.uint8)
        img_bg = np.full((3, 3, 1), 255, dtype=np.uint8)
        img_blend = blend.blend_alpha(img_fg, img_bg, 0.0, eps=0)
        assert img_blend.dtype.name == "uint8"
        assert img_blend.shape == (3, 3, 1)
        assert np.all(img_blend == 255)

    def test_alpha_is_0_2d_arrays(self):
        img_fg = np.full((3, 3), 0, dtype=np.uint8)
        img_bg = np.full((3, 3), 255, dtype=np.uint8)
        img_blend = blend.blend_alpha(img_fg, img_bg, 0.0, eps=0)
        assert img_blend.dtype.name == "uint8"
        assert img_blend.shape == (3, 3)
        assert np.all(img_blend == 255)

    def test_alpha_is_030(self):
        img_fg = np.full((3, 3, 1), 0, dtype=np.uint8)
        img_bg = np.full((3, 3, 1), 255, dtype=np.uint8)
        img_blend = blend.blend_alpha(img_fg, img_bg, 0.3, eps=0)
        assert img_blend.dtype.name == "uint8"
        assert img_blend.shape == (3, 3, 1)
        assert np.allclose(img_blend, 0.7*255, atol=1.01, rtol=0)

    def test_alpha_is_030_2d_arrays(self):
        img_fg = np.full((3, 3), 0, dtype=np.uint8)
        img_bg = np.full((3, 3), 255, dtype=np.uint8)
        img_blend = blend.blend_alpha(img_fg, img_bg, 0.3, eps=0)
        assert img_blend.dtype.name == "uint8"
        assert img_blend.shape == (3, 3)
        assert np.allclose(img_blend, 0.7*255, atol=1.01, rtol=0)

    def test_channelwise_alpha(self):
        img_fg = np.full((3, 3, 2), 0, dtype=np.uint8)
        img_bg = np.full((3, 3, 2), 255, dtype=np.uint8)
        img_blend = blend.blend_alpha(img_fg, img_bg, [1.0, 0.0], eps=0)
        assert img_blend.dtype.name == "uint8"
        assert img_blend.shape == (3, 3, 2)
        assert np.all(img_blend[:, :, 0] == 0)
        assert np.all(img_blend[:, :, 1] == 255)

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image_fg = np.full(shape, 0, dtype=np.uint8)
                image_bg = np.full(shape, 255, dtype=np.uint8)

                image_aug = blend.blend_alpha(image_fg, image_bg, 1.0)

                assert np.all(image_aug == 0)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image_fg = np.full(shape, 0, dtype=np.uint8)
                image_bg = np.full(shape, 255, dtype=np.uint8)

                image_aug = blend.blend_alpha(image_fg, image_bg, 1.0)

                assert np.all(image_aug == 0)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_other_dtypes_bool(self):
        img_fg = np.full((3, 3, 1), 0, dtype=bool)
        img_bg = np.full((3, 3, 1), 1, dtype=bool)
        img_blend = blend.blend_alpha(img_fg, img_bg, 0.3, eps=0)
        assert img_blend.dtype.name == "bool"
        assert img_blend.shape == (3, 3, 1)
        assert np.all(img_blend == 1)

    def test_other_dtypes_bool_2d_arrays(self):
        img_fg = np.full((3, 3), 0, dtype=bool)
        img_bg = np.full((3, 3), 1, dtype=bool)
        img_blend = blend.blend_alpha(img_fg, img_bg, 0.3, eps=0)
        assert img_blend.dtype.name == "bool"
        assert img_blend.shape == (3, 3)
        assert np.all(img_blend == 1)

    # TODO split this up into multiple tests
    def test_other_dtypes_uint_int(self):
        dtypes = ["uint8", "uint16", "uint32", "uint64",
                  "int8", "int16", "int32", "int64"]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                dtype = np.dtype(dtype)

                min_value, center_value, max_value = \
                    iadt.get_value_range_of_dtype(dtype)

                values = [
                    (0, 0),
                    (0, 10),
                    (10, 20),
                    (min_value, min_value),
                    (max_value, max_value),
                    (min_value, max_value),
                    (min_value, int(center_value)),
                    (int(center_value), max_value),
                    (int(center_value + 0.20 * max_value), max_value),
                    (int(center_value + 0.27 * max_value), max_value),
                    (int(center_value + 0.40 * max_value), max_value),
                    (min_value, 0),
                    (0, max_value)
                ]
                values = values + [(v2, v1) for v1, v2 in values]

                for v1, v2 in values:
                    v1_scalar = np.full((), v1, dtype=dtype)
                    v2_scalar = np.full((), v2, dtype=dtype)
                    
                    img_fg = np.full((3, 3, 1), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 1), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(img_fg, img_bg, 1.0, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 1)
                    assert np.all(img_blend == v1_scalar)

                    img_fg = np.full((3, 3, 1), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 1), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(img_fg, img_bg, 0.99, eps=0.1)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 1)
                    assert np.all(img_blend == v1_scalar)

                    img_fg = np.full((3, 3, 1), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 1), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(img_fg, img_bg, 0.0, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 1)
                    assert np.all(img_blend == v2_scalar)

                    # TODO this test breaks for numpy <1.15 -- why?
                    for c in sm.xrange(3):
                        img_fg = np.full((3, 3, c), v1, dtype=dtype)
                        img_bg = np.full((3, 3, c), v2, dtype=dtype)
                        img_blend = blend.blend_alpha(
                            img_fg, img_bg, 0.75, eps=0)
                        assert img_blend.dtype.name == np.dtype(dtype)
                        assert img_blend.shape == (3, 3, c)
                        for ci in sm.xrange(c):
                            v_blend = min(
                                max(
                                    int(
                                        0.75*np.float128(v1)
                                        + 0.25*np.float128(v2)
                                    ),
                                    min_value
                                ),
                                max_value)
                            diff = (
                                v_blend - img_blend
                                if v_blend > img_blend[0, 0, ci]
                                else img_blend - v_blend)
                            assert np.all(diff < 1.01)

                    img_fg = np.full((3, 3, 2), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 2), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(img_fg, img_bg, 0.75, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 2)
                    v_blend = min(
                        max(
                            int(
                                0.75 * np.float128(v1)
                                + 0.25 * np.float128(v2)
                            ),
                            min_value
                        ),
                        max_value)
                    diff = (
                        v_blend - img_blend
                        if v_blend > img_blend[0, 0, 0]
                        else img_blend - v_blend
                    )
                    assert np.all(diff < 1.01)

                    img_fg = np.full((3, 3, 2), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 2), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(
                        img_fg, img_bg, [1.0, 0.0], eps=0.1)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 2)
                    assert np.all(img_blend[:, :, 0] == v1_scalar)
                    assert np.all(img_blend[:, :, 1] == v2_scalar)

                    # elementwise, alphas.shape = (1, 2)
                    img_fg = np.full((1, 2, 3), v1, dtype=dtype)
                    img_bg = np.full((1, 2, 3), v2, dtype=dtype)
                    alphas = np.zeros((1, 2), dtype=np.float64)
                    alphas[:, :] = [1.0, 0.0]
                    img_blend = blend.blend_alpha(img_fg, img_bg, alphas, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (1, 2, 3)
                    assert np.all(img_blend[0, 0, :] == v1_scalar)
                    assert np.all(img_blend[0, 1, :] == v2_scalar)

                    # elementwise, alphas.shape = (1, 2, 1)
                    img_fg = np.full((1, 2, 3), v1, dtype=dtype)
                    img_bg = np.full((1, 2, 3), v2, dtype=dtype)
                    alphas = np.zeros((1, 2, 1), dtype=np.float64)
                    alphas[:, :, 0] = [1.0, 0.0]
                    img_blend = blend.blend_alpha(img_fg, img_bg, alphas, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (1, 2, 3)
                    assert np.all(img_blend[0, 0, :] == v1_scalar)
                    assert np.all(img_blend[0, 1, :] == v2_scalar)

                    # elementwise, alphas.shape = (1, 2, 3)
                    img_fg = np.full((1, 2, 3), v1, dtype=dtype)
                    img_bg = np.full((1, 2, 3), v2, dtype=dtype)
                    alphas = np.zeros((1, 2, 3), dtype=np.float64)
                    alphas[:, :, 0] = [1.0, 0.0]
                    alphas[:, :, 1] = [0.0, 1.0]
                    alphas[:, :, 2] = [1.0, 0.0]
                    img_blend = blend.blend_alpha(img_fg, img_bg, alphas, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (1, 2, 3)
                    assert np.all(img_blend[0, 0, [0, 2]] == v1_scalar)
                    assert np.all(img_blend[0, 1, [0, 2]] == v2_scalar)
                    assert np.all(img_blend[0, 0, 1] == v2_scalar)
                    assert np.all(img_blend[0, 1, 1] == v1_scalar)

    # TODO split this up into multiple tests
    def test_other_dtypes_float(self):
        dtypes = ["float16", "float32", "float64"]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                dtype = np.dtype(dtype)

                def _allclose(a, b):
                    atol = 1e-4 if dtype == np.float16 else 1e-8
                    return np.allclose(a, b, atol=atol, rtol=0)

                isize = np.dtype(dtype).itemsize
                max_value = 1000 ** (isize - 1)
                min_value = -max_value
                center_value = 0
                values = [
                    (0, 0),
                    (0, 10),
                    (10, 20),
                    (min_value, min_value),
                    (max_value, max_value),
                    (min_value, max_value),
                    (min_value, center_value),
                    (center_value, max_value),
                    (center_value + 0.20 * max_value, max_value),
                    (center_value + 0.27 * max_value, max_value),
                    (center_value + 0.40 * max_value, max_value),
                    (min_value, 0),
                    (0, max_value)
                ]
                values = values + [(v2, v1) for v1, v2 in values]

                max_float_dt = np.float128

                for v1, v2 in values:
                    v1_scalar = np.full((), v1, dtype=dtype)
                    v2_scalar = np.full((), v2, dtype=dtype)

                    img_fg = np.full((3, 3, 1), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 1), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(img_fg, img_bg, 1.0, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 1)
                    assert _allclose(img_blend, max_float_dt(v1))

                    img_fg = np.full((3, 3, 1), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 1), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(
                        img_fg, img_bg, 0.99, eps=0.1)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 1)
                    assert _allclose(img_blend, max_float_dt(v1))

                    img_fg = np.full((3, 3, 1), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 1), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(img_fg, img_bg, 0.0, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 1)
                    assert _allclose(img_blend, max_float_dt(v2))

                    for c in sm.xrange(3):
                        img_fg = np.full((3, 3, c), v1, dtype=dtype)
                        img_bg = np.full((3, 3, c), v2, dtype=dtype)
                        img_blend = blend.blend_alpha(
                            img_fg, img_bg, 0.75, eps=0)
                        assert img_blend.dtype.name == np.dtype(dtype)
                        assert img_blend.shape == (3, 3, c)
                        assert _allclose(
                            img_blend,
                            0.75*max_float_dt(v1) + 0.25*max_float_dt(v2)
                        )

                    img_fg = np.full((3, 3, 2), v1, dtype=dtype)
                    img_bg = np.full((3, 3, 2), v2, dtype=dtype)
                    img_blend = blend.blend_alpha(
                        img_fg, img_bg, [1.0, 0.0], eps=0.1)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (3, 3, 2)
                    assert _allclose(img_blend[:, :, 0], max_float_dt(v1))
                    assert _allclose(img_blend[:, :, 1], max_float_dt(v2))

                    # elementwise, alphas.shape = (1, 2)
                    img_fg = np.full((1, 2, 3), v1, dtype=dtype)
                    img_bg = np.full((1, 2, 3), v2, dtype=dtype)
                    alphas = np.zeros((1, 2), dtype=np.float64)
                    alphas[:, :] = [1.0, 0.0]
                    img_blend = blend.blend_alpha(
                        img_fg, img_bg, alphas, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (1, 2, 3)
                    assert _allclose(img_blend[0, 0, :], v1_scalar)
                    assert _allclose(img_blend[0, 1, :], v2_scalar)

                    # elementwise, alphas.shape = (1, 2, 1)
                    img_fg = np.full((1, 2, 3), v1, dtype=dtype)
                    img_bg = np.full((1, 2, 3), v2, dtype=dtype)
                    alphas = np.zeros((1, 2, 1), dtype=np.float64)
                    alphas[:, :, 0] = [1.0, 0.0]
                    img_blend = blend.blend_alpha(
                        img_fg, img_bg, alphas, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (1, 2, 3)
                    assert _allclose(img_blend[0, 0, :], v1_scalar)
                    assert _allclose(img_blend[0, 1, :], v2_scalar)

                    # elementwise, alphas.shape = (1, 2, 3)
                    img_fg = np.full((1, 2, 3), v1, dtype=dtype)
                    img_bg = np.full((1, 2, 3), v2, dtype=dtype)
                    alphas = np.zeros((1, 2, 3), dtype=np.float64)
                    alphas[:, :, 0] = [1.0, 0.0]
                    alphas[:, :, 1] = [0.0, 1.0]
                    alphas[:, :, 2] = [1.0, 0.0]
                    img_blend = blend.blend_alpha(
                        img_fg, img_bg, alphas, eps=0)
                    assert img_blend.dtype.name == np.dtype(dtype)
                    assert img_blend.shape == (1, 2, 3)
                    assert _allclose(img_blend[0, 0, [0, 2]], v1_scalar)
                    assert _allclose(img_blend[0, 1, [0, 2]], v2_scalar)
                    assert _allclose(img_blend[0, 0, 1], v2_scalar)
                    assert _allclose(img_blend[0, 1, 1], v1_scalar)


class TestAlpha(unittest.TestCase):
    def test_deprecation_warning(self):
        aug1 = iaa.Sequential([])
        aug2 = iaa.Sequential([])

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            aug = iaa.Alpha(0.75, first=aug1, second=aug2)

            assert (
                "is deprecated"
                in str(caught_warnings[-1].message)
            )

        assert isinstance(aug, iaa.BlendAlpha)
        assert np.isclose(aug.factor.value, 0.75)
        assert aug.foreground is aug1
        assert aug.background is aug2


class TestBlendAlpha(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        base_img = np.zeros((3, 3, 1), dtype=np.uint8)
        return base_img

    @property
    def heatmaps(self):
        heatmaps_arr = np.float32([[0.0, 0.0, 1.0],
                                   [0.0, 0.0, 1.0],
                                   [0.0, 1.0, 1.0]])
        return HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    @property
    def heatmaps_r1(self):
        heatmaps_arr_r1 = np.float32([[0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0]])
        return HeatmapsOnImage(heatmaps_arr_r1, shape=(3, 3, 3))

    @property
    def heatmaps_l1(self):
        heatmaps_arr_l1 = np.float32([[0.0, 1.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [1.0, 1.0, 0.0]])
        return HeatmapsOnImage(heatmaps_arr_l1, shape=(3, 3, 3))

    @property
    def segmaps(self):
        segmaps_arr = np.int32([[0, 0, 1],
                                [0, 0, 1],
                                [0, 1, 1]])
        return SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))

    @property
    def segmaps_r1(self):
        segmaps_arr_r1 = np.int32([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 1]])
        return SegmentationMapsOnImage(segmaps_arr_r1, shape=(3, 3, 3))

    @property
    def segmaps_l1(self):
        segmaps_arr_l1 = np.int32([[0, 1, 0],
                                   [0, 1, 0],
                                   [1, 1, 0]])
        return SegmentationMapsOnImage(segmaps_arr_l1, shape=(3, 3, 3))

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=5, y=10), ia.Keypoint(x=6, y=11)]
        return ia.KeypointsOnImage(kps, shape=(20, 20, 3))

    @property
    def psoi(self):
        ps = [ia.Polygon([(5, 5), (10, 5), (10, 10)])]
        return ia.PolygonsOnImage(ps, shape=(20, 20, 3))

    @property
    def lsoi(self):
        lss = [ia.LineString([(5, 5), (10, 5), (10, 10)])]
        return ia.LineStringsOnImage(lss, shape=(20, 20, 3))

    @property
    def bbsoi(self):
        bbs = [ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)]
        return ia.BoundingBoxesOnImage(bbs, shape=(20, 20, 3))

    def test_images_factor_is_1(self):
        aug = iaa.BlendAlpha(1, iaa.Add(10), iaa.Add(20))
        observed = aug.augment_image(self.image)
        expected = np.round(self.image + 10).astype(np.uint8)
        assert np.allclose(observed, expected)

    def test_heatmaps_factor_is_1_with_affines_and_per_channel(self):
        for per_channel in [False, True]:
            with self.subTest(per_channel=per_channel):
                aug = iaa.BlendAlpha(
                    1,
                    iaa.Affine(translate_px={"x": 1}),
                    iaa.Affine(translate_px={"x": -1}),
                    per_channel=per_channel)
                observed = aug.augment_heatmaps([self.heatmaps])[0]
                assert observed.shape == self.heatmaps.shape
                assert 0 - 1e-6 < self.heatmaps.min_value < 0 + 1e-6
                assert 1 - 1e-6 < self.heatmaps.max_value < 1 + 1e-6
                assert np.allclose(observed.get_arr(),
                                   self.heatmaps_r1.get_arr())

    def test_segmaps_factor_is_1_with_affines_and_per_channel(self):
        for per_channel in [False, True]:
            with self.subTest(per_channel=per_channel):
                aug = iaa.BlendAlpha(
                    1,
                    iaa.Affine(translate_px={"x": 1}),
                    iaa.Affine(translate_px={"x": -1}),
                    per_channel=per_channel)
                observed = aug.augment_segmentation_maps([self.segmaps])[0]
                assert observed.shape == self.segmaps.shape
                assert np.array_equal(observed.get_arr(),
                                      self.segmaps_r1.get_arr())

    def test_images_factor_is_0(self):
        aug = iaa.BlendAlpha(0, iaa.Add(10), iaa.Add(20))
        observed = aug.augment_image(self.image)
        expected = np.round(self.image + 20).astype(np.uint8)
        assert np.allclose(observed, expected)

    def test_heatmaps_factor_is_0_with_affines_and_per_channel(self):
        for per_channel in [False, True]:
            with self.subTest(per_channel=per_channel):
                aug = iaa.BlendAlpha(
                    0,
                    iaa.Affine(translate_px={"x": 1}),
                    iaa.Affine(translate_px={"x": -1}),
                    per_channel=per_channel)
                observed = aug.augment_heatmaps([self.heatmaps])[0]
                assert observed.shape == self.heatmaps.shape
                assert 0 - 1e-6 < self.heatmaps.min_value < 0 + 1e-6
                assert 1 - 1e-6 < self.heatmaps.max_value < 1 + 1e-6
                assert np.allclose(observed.get_arr(),
                                   self.heatmaps_l1.get_arr())

    def test_segmaps_factor_is_0_with_affines_and_per_channel(self):
        for per_channel in [False, True]:
            with self.subTest(per_channel=per_channel):
                aug = iaa.BlendAlpha(
                    0,
                    iaa.Affine(translate_px={"x": 1}),
                    iaa.Affine(translate_px={"x": -1}),
                    per_channel=per_channel)
                observed = aug.augment_segmentation_maps([self.segmaps])[0]
                assert observed.shape == self.segmaps.shape
                assert np.array_equal(observed.get_arr(),
                                      self.segmaps_l1.get_arr())

    def test_images_factor_is_075(self):
        aug = iaa.BlendAlpha(0.75, iaa.Add(10), iaa.Add(20))
        observed = aug.augment_image(self.image)
        expected = np.round(
            self.image
            + 0.75 * 10
            + 0.25 * 20
        ).astype(np.uint8)
        assert np.allclose(observed, expected)

    def test_images_factor_is_075_fg_branch_is_none(self):
        aug = iaa.BlendAlpha(0.75, None, iaa.Add(20))
        observed = aug.augment_image(self.image + 10)
        expected = np.round(
            self.image
            + 0.75 * 10
            + 0.25 * (10 + 20)
        ).astype(np.uint8)
        assert np.allclose(observed, expected)

    def test_images_factor_is_075_bg_branch_is_none(self):
        aug = iaa.BlendAlpha(0.75, iaa.Add(10), None)
        observed = aug.augment_image(self.image + 10)
        expected = np.round(
            self.image
            + 0.75 * (10 + 10)
            + 0.25 * 10
        ).astype(np.uint8)
        assert np.allclose(observed, expected)

    def test_images_factor_is_tuple(self):
        image = np.zeros((1, 2, 1), dtype=np.uint8)
        nb_iterations = 1000
        aug = iaa.BlendAlpha((0.0, 1.0), iaa.Add(10), iaa.Add(110))
        values = []
        for _ in sm.xrange(nb_iterations):
            observed = aug.augment_image(image)
            observed_val = np.round(np.average(observed)) - 10
            values.append(observed_val / 100)

        nb_bins = 5
        hist, _ = np.histogram(values, bins=nb_bins, range=(0.0, 1.0),
                               density=False)
        density_expected = 1.0/nb_bins
        density_tolerance = 0.05
        for nb_samples in hist:
            density = nb_samples / nb_iterations
            assert np.isclose(density, density_expected,
                              rtol=0, atol=density_tolerance)

    def test_bad_datatype_for_factor_fails(self):
        got_exception = False
        try:
            _ = iaa.BlendAlpha(False, iaa.Add(10), None)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_images_with_per_channel_in_both_alpha_and_child(self):
        image = np.zeros((1, 1, 1000), dtype=np.uint8)
        aug = iaa.BlendAlpha(
            1.0,
            iaa.Add((0, 100), per_channel=True),
            None,
            per_channel=True)
        observed = aug.augment_image(image)
        uq = np.unique(observed)
        assert len(uq) > 1
        assert np.max(observed) > 80
        assert np.min(observed) < 20

    def test_images_with_per_channel_in_alpha_and_tuple_as_factor(self):
        image = np.zeros((1, 1, 1000), dtype=np.uint8)
        aug = iaa.BlendAlpha(
            (0.0, 1.0),
            iaa.Add(100),
            None,
            per_channel=True)
        observed = aug.augment_image(image)
        uq = np.unique(observed)
        assert len(uq) > 1
        assert np.max(observed) > 80
        assert np.min(observed) < 20

    def test_images_float_as_per_channel_tuple_as_factor_two_branches(self):
        aug = iaa.BlendAlpha(
            (0.0, 1.0),
            iaa.Add(100),
            iaa.Add(0),
            per_channel=0.5)
        seen = [0, 0]
        for _ in sm.xrange(200):
            observed = aug.augment_image(np.zeros((1, 1, 100), dtype=np.uint8))
            uq = np.unique(observed)
            if len(uq) == 1:
                seen[0] += 1
            elif len(uq) > 1:
                seen[1] += 1
            else:
                assert False
        assert 100 - 50 < seen[0] < 100 + 50
        assert 100 - 50 < seen[1] < 100 + 50

    def test_bad_datatype_for_per_channel_fails(self):
        # bad datatype for per_channel
        got_exception = False
        try:
            _ = iaa.BlendAlpha(0.5, iaa.Add(10), None, per_channel="test")
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_hooks_limiting_propagation(self):
        aug = iaa.BlendAlpha(0.5, iaa.Add(100), iaa.Add(50), name="AlphaTest")

        def propagator(images, augmenter, parents, default):
            if "Alpha" in augmenter.name:
                return False
            else:
                return default

        hooks = ia.HooksImages(propagator=propagator)
        image = np.zeros((10, 10, 3), dtype=np.uint8) + 1
        observed = aug.augment_image(image, hooks=hooks)
        assert np.array_equal(observed, image)

    def test_keypoints_factor_is_1(self):
        self._test_cba_factor_is_1("augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_0501(self):
        self._test_cba_factor_is_0501("augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_0(self):
        self._test_cba_factor_is_0("augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_0499(self):
        self._test_cba_factor_is_0499("augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_1_with_per_channel(self):
        self._test_cba_factor_is_1_and_per_channel(
            "augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_0_with_per_channel(self):
        self._test_cba_factor_is_0_and_per_channel(
            "augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_choice_of_vals_close_to_050_per_channel(self):
        self._test_cba_factor_is_choice_around_050_and_per_channel(
            "augment_keypoints", self.kpsoi)

    def test_keypoints_are_empty(self):
        self._test_empty_cba(
            "augment_keypoints", ia.KeypointsOnImage([], shape=(1, 2, 3)))

    def test_keypoints_hooks_limit_propagation(self):
        self._test_cba_hooks_limit_propagation(
            "augment_keypoints", self.kpsoi)

    def test_polygons_factor_is_1(self):
        self._test_cba_factor_is_1("augment_polygons", self.psoi)

    def test_polygons_factor_is_0501(self):
        self._test_cba_factor_is_0501("augment_polygons", self.psoi)

    def test_polygons_factor_is_0(self):
        self._test_cba_factor_is_0("augment_polygons", self.psoi)

    def test_polygons_factor_is_0499(self):
        self._test_cba_factor_is_0499("augment_polygons", self.psoi)

    def test_polygons_factor_is_1_and_per_channel(self):
        self._test_cba_factor_is_1_and_per_channel(
            "augment_polygons", self.psoi)

    def test_polygons_factor_is_0_and_per_channel(self):
        self._test_cba_factor_is_0_and_per_channel(
            "augment_polygons", self.psoi)

    def test_polygons_factor_is_choice_around_050_and_per_channel(self):
        self._test_cba_factor_is_choice_around_050_and_per_channel(
            "augment_polygons", self.psoi
        )

    def test_empty_polygons(self):
        return self._test_empty_cba(
            "augment_polygons", ia.PolygonsOnImage([], shape=(1, 2, 3)))

    def test_polygons_hooks_limit_propagation(self):
        return self._test_cba_hooks_limit_propagation(
            "augment_polygons", self.psoi)

    def test_line_strings_factor_is_1(self):
        self._test_cba_factor_is_1("augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_0501(self):
        self._test_cba_factor_is_0501("augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_0(self):
        self._test_cba_factor_is_0("augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_0499(self):
        self._test_cba_factor_is_0499("augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_1_and_per_channel(self):
        self._test_cba_factor_is_1_and_per_channel(
            "augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_0_and_per_channel(self):
        self._test_cba_factor_is_0_and_per_channel(
            "augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_choice_around_050_and_per_channel(self):
        self._test_cba_factor_is_choice_around_050_and_per_channel(
            "augment_line_strings", self.lsoi
        )

    def test_empty_line_strings(self):
        return self._test_empty_cba(
            "augment_line_strings",
            ia.LineStringsOnImage([], shape=(1, 2, 3)))

    def test_line_strings_hooks_limit_propagation(self):
        return self._test_cba_hooks_limit_propagation(
            "augment_line_strings", self.lsoi)

    def test_bounding_boxes_factor_is_1(self):
        self._test_cba_factor_is_1("augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_0501(self):
        self._test_cba_factor_is_0501("augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_0(self):
        self._test_cba_factor_is_0("augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_0499(self):
        self._test_cba_factor_is_0499("augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_1_and_per_channel(self):
        self._test_cba_factor_is_1_and_per_channel(
            "augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_0_and_per_channel(self):
        self._test_cba_factor_is_0_and_per_channel(
            "augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_choice_around_050_and_per_channel(self):
        self._test_cba_factor_is_choice_around_050_and_per_channel(
            "augment_bounding_boxes", self.bbsoi
        )

    def test_empty_bounding_boxes(self):
        return self._test_empty_cba(
            "augment_bounding_boxes",
            ia.BoundingBoxesOnImage([], shape=(1, 2, 3)))

    def test_bounding_boxes_hooks_limit_propagation(self):
        return self._test_cba_hooks_limit_propagation(
            "augment_bounding_boxes", self.bbsoi)

    # Tests for CBA (=coordinate based augmentable) below. This currently
    # covers keypoints, polygons and bounding boxes.

    @classmethod
    def _test_cba_factor_is_1(cls, augf_name, cbaoi):
        aug = iaa.BlendAlpha(
            1.0, iaa.Identity(), iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)([cbaoi])

        assert_cbaois_equal(observed[0], cbaoi)

    @classmethod
    def _test_cba_factor_is_0501(cls, augf_name, cbaoi):
        aug = iaa.BlendAlpha(0.501, iaa.Identity(),
                        iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)([cbaoi])

        assert_cbaois_equal(observed[0], cbaoi)

    @classmethod
    def _test_cba_factor_is_0(cls, augf_name, cbaoi):
        aug = iaa.BlendAlpha(
            0.0, iaa.Identity(), iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)([cbaoi])

        expected = shift_cbaoi(cbaoi, left=1)
        assert_cbaois_equal(observed[0], expected)

    @classmethod
    def _test_cba_factor_is_0499(cls, augf_name, cbaoi):
        aug = iaa.BlendAlpha(0.499, iaa.Identity(),
                        iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)([cbaoi])

        expected = shift_cbaoi(cbaoi, left=1)
        assert_cbaois_equal(observed[0], expected)

    @classmethod
    def _test_cba_factor_is_1_and_per_channel(cls, augf_name, cbaoi):
        aug = iaa.BlendAlpha(
            1.0,
            iaa.Identity(),
            iaa.Affine(translate_px={"x": 1}),
            per_channel=True)

        observed = getattr(aug, augf_name)([cbaoi])

        assert_cbaois_equal(observed[0], cbaoi)

    @classmethod
    def _test_cba_factor_is_0_and_per_channel(cls, augf_name, cbaoi):
        aug = iaa.BlendAlpha(
            0.0,
            iaa.Identity(),
            iaa.Affine(translate_px={"x": 1}),
            per_channel=True)

        observed = getattr(aug, augf_name)([cbaoi])

        expected = shift_cbaoi(cbaoi, left=1)
        assert_cbaois_equal(observed[0], expected)

    @classmethod
    def _test_cba_factor_is_choice_around_050_and_per_channel(
            cls, augf_name, cbaoi):
        aug = iaa.BlendAlpha(
            iap.Choice([0.49, 0.51]),
            iaa.Identity(),
            iaa.Affine(translate_px={"x": 1}),
            per_channel=True)
        expected_same = cbaoi.deepcopy()
        expected_shifted = shift_cbaoi(cbaoi, left=1)
        seen = [0, 0, 0]
        for _ in sm.xrange(200):
            observed = getattr(aug, augf_name)([cbaoi])[0]

            assert len(observed.items) == len(expected_same.items)
            assert len(observed.items) == len(expected_shifted.items)

            # We use here allclose() instead of coords_almost_equals()
            # as the latter one is much slower for polygons and we don't have
            # to deal with tricky geometry changes here, just naive shifting.
            if np.allclose(observed.items[0].coords,
                           expected_same.items[0].coords,
                           rtol=0, atol=0.1):
                seen[0] += 1
            elif np.allclose(observed.items[0].coords,
                             expected_shifted.items[0].coords,
                             rtol=0, atol=0.1):
                seen[1] += 1
            else:
                seen[2] += 1
        assert 100 - 50 < seen[0] < 100 + 50
        assert 100 - 50 < seen[1] < 100 + 50
        assert seen[2] == 0

    @classmethod
    def _test_empty_cba(cls, augf_name, cbaoi):
        # empty CBAs
        aug = iaa.BlendAlpha(0.501, iaa.Identity(),
                        iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)(cbaoi)

        assert len(observed.items) == 0
        assert observed.shape == cbaoi.shape

    @classmethod
    def _test_cba_hooks_limit_propagation(cls, augf_name, cbaoi):
        aug = iaa.BlendAlpha(
            0.0,
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"y": 1}),
            name="AlphaTest")

        def propagator(cbaoi_to_aug, augmenter, parents, default):
            if "Alpha" in augmenter.name:
                return False
            else:
                return default

        # no hooks for polygons yet, so we use HooksKeypoints
        hooks = ia.HooksKeypoints(propagator=propagator)
        observed = getattr(aug, augf_name)([cbaoi], hooks=hooks)[0]
        assert observed.items[0].coords_almost_equals(cbaoi.items[0])

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 0, dtype=np.uint8)
                aug = iaa.BlendAlpha(1.0, iaa.Add(1), iaa.Add(100))

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 0, dtype=np.uint8)
                aug = iaa.BlendAlpha(1.0, iaa.Add(1), iaa.Add(100))

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_get_parameters(self):
        fg = iaa.Identity()
        bg = iaa.Sequential([iaa.Add(1)])
        aug = iaa.BlendAlpha(0.65, fg, bg, per_channel=1)
        params = aug.get_parameters()
        assert isinstance(params[0], iap.Deterministic)
        assert isinstance(params[1], iap.Deterministic)
        assert 0.65 - 1e-6 < params[0].value < 0.65 + 1e-6
        assert params[1].value == 1

    def test_get_children_lists(self):
        fg = iaa.Identity()
        bg = iaa.Sequential([iaa.Add(1)])
        aug = iaa.BlendAlpha(0.65, fg, bg, per_channel=1)
        children_lsts = aug.get_children_lists()
        assert len(children_lsts) == 2
        assert ia.is_iterable([lst for lst in children_lsts])
        assert fg in children_lsts[0]
        assert bg == children_lsts[1]

    def test_to_deterministic(self):
        class _DummyAugmenter(iaa.Identity):
            def __init__(self, *args, **kwargs):
                super(_DummyAugmenter, self).__init__(*args, **kwargs)
                self.deterministic_called = False

            def _to_deterministic(self):
                self.deterministic_called = True
                return self

        identity1 = _DummyAugmenter()
        identity2 = _DummyAugmenter()
        aug = iaa.BlendAlpha(0.5, identity1, identity2)

        aug_det = aug.to_deterministic()

        assert aug_det.deterministic
        assert aug_det.random_state is not aug.random_state
        assert aug_det.foreground.deterministic
        assert aug_det.background.deterministic
        assert identity1.deterministic_called is True
        assert identity2.deterministic_called is True

    def test_pickleable(self):
        aug = iaa.BlendAlpha(
            (0.1, 0.9),
            iaa.Add((1, 10), random_state=1),
            iaa.Add((11, 20), random_state=2),
            per_channel=True,
            random_state=3)
        runtest_pickleable_uint8_img(aug, iterations=10)


class _DummyMaskParameter(iap.StochasticParameter):
    def __init__(self, inverted=False):
        super(_DummyMaskParameter, self).__init__()
        self.inverted = inverted

    def _draw_samples(self, size, random_state):
        h, w = size[0:2]
        nb_channels = 1 if len(size) == 2 else size[2]
        assert nb_channels <= 3
        result = []
        for i in np.arange(nb_channels):
            if i == 0:
                result.append(np.zeros((h, w), dtype=np.float32))
            else:
                result.append(np.ones((h, w), dtype=np.float32))
        result = np.stack(result, axis=-1)
        if len(size) == 2:
            result = result[:, :, 0]
        if self.inverted:
            result = 1.0 - result
        return result


class TestAlphaElementwise(unittest.TestCase):
    def test_deprecation_warning(self):
        aug1 = iaa.Sequential([])
        aug2 = iaa.Sequential([])

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            aug = iaa.AlphaElementwise(factor=0.5, first=aug1, second=aug2)

            assert (
                "is deprecated"
                in str(caught_warnings[-1].message)
            )

        assert isinstance(aug, iaa.BlendAlphaElementwise)
        assert np.isclose(aug.factor.value, 0.5)
        assert aug.foreground is aug1
        assert aug.background is aug2


# TODO add tests for heatmaps and segmaps that differ from the image size
class TestBlendAlphaElementwise(unittest.TestCase):
    def setUp(self):
        reseed()

    @property
    def image(self):
        base_img = np.zeros((3, 3, 1), dtype=np.uint8)
        return base_img

    @property
    def heatmaps(self):
        heatmaps_arr = np.float32([[0.0, 0.0, 1.0],
                                   [0.0, 0.0, 1.0],
                                   [0.0, 1.0, 1.0]])
        return HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    @property
    def heatmaps_r1(self):
        heatmaps_arr_r1 = np.float32([[0.0, 0.0, 0.0],
                                      [0.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0]])
        return HeatmapsOnImage(heatmaps_arr_r1, shape=(3, 3, 3))

    @property
    def heatmaps_l1(self):
        heatmaps_arr_l1 = np.float32([[0.0, 1.0, 0.0],
                                      [0.0, 1.0, 0.0],
                                      [1.0, 1.0, 0.0]])

        return HeatmapsOnImage(heatmaps_arr_l1, shape=(3, 3, 3))

    @property
    def segmaps(self):
        segmaps_arr = np.int32([[0, 0, 1],
                                [0, 0, 1],
                                [0, 1, 1]])
        return SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))

    @property
    def segmaps_r1(self):
        segmaps_arr_r1 = np.int32([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 1]])
        return SegmentationMapsOnImage(segmaps_arr_r1, shape=(3, 3, 3))

    @property
    def segmaps_l1(self):
        segmaps_arr_l1 = np.int32([[0, 1, 0],
                                   [0, 1, 0],
                                   [1, 1, 0]])
        return SegmentationMapsOnImage(segmaps_arr_l1, shape=(3, 3, 3))

    @property
    def kpsoi(self):
        kps = [ia.Keypoint(x=5, y=10), ia.Keypoint(x=6, y=11)]
        return ia.KeypointsOnImage(kps, shape=(20, 20, 3))

    @property
    def psoi(self):
        ps = [ia.Polygon([(5, 5), (10, 5), (10, 10)])]
        return ia.PolygonsOnImage(ps, shape=(20, 20, 3))

    @property
    def lsoi(self):
        lss = [ia.LineString([(5, 5), (10, 5), (10, 10)])]
        return ia.LineStringsOnImage(lss, shape=(20, 20, 3))

    @property
    def bbsoi(self):
        bbs = [ia.BoundingBox(x1=5, y1=6, x2=7, y2=8)]
        return ia.BoundingBoxesOnImage(bbs, shape=(20, 20, 3))

    def test_images_factor_is_1(self):
        aug = iaa.BlendAlphaElementwise(1, iaa.Add(10), iaa.Add(20))
        observed = aug.augment_image(self.image)
        expected = self.image + 10
        assert np.allclose(observed, expected)

    def test_heatmaps_factor_is_1_with_affines(self):
        aug = iaa.BlendAlphaElementwise(
            1,
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"x": -1}))
        observed = aug.augment_heatmaps([self.heatmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps_r1.get_arr())

    def test_segmaps_factor_is_1_with_affines(self):
        aug = iaa.BlendAlphaElementwise(
            1,
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"x": -1}))
        observed = aug.augment_segmentation_maps([self.segmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert np.array_equal(observed.get_arr(), self.segmaps_r1.get_arr())

    def test_images_factor_is_0(self):
        aug = iaa.BlendAlphaElementwise(0, iaa.Add(10), iaa.Add(20))
        observed = aug.augment_image(self.image)
        expected = self.image + 20
        assert np.allclose(observed, expected)

    def test_heatmaps_factor_is_0_with_affines(self):
        aug = iaa.BlendAlphaElementwise(
            0,
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"x": -1}))
        observed = aug.augment_heatmaps([self.heatmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps_l1.get_arr())

    def test_segmaps_factor_is_0_with_affines(self):
        aug = iaa.BlendAlphaElementwise(
            0,
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"x": -1}))
        observed = aug.augment_segmentation_maps([self.segmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert np.array_equal(observed.get_arr(), self.segmaps_l1.get_arr())

    def test_images_factor_is_075(self):
        aug = iaa.BlendAlphaElementwise(0.75, iaa.Add(10), iaa.Add(20))
        observed = aug.augment_image(self.image)
        expected = np.round(
            self.image + 0.75 * 10 + 0.25 * 20
        ).astype(np.uint8)
        assert np.allclose(observed, expected)

    def test_images_factor_is_075_fg_branch_is_none(self):
        aug = iaa.BlendAlphaElementwise(0.75, None, iaa.Add(20))
        observed = aug.augment_image(self.image + 10)
        expected = np.round(
            self.image + 0.75 * 10 + 0.25 * (10 + 20)
        ).astype(np.uint8)
        assert np.allclose(observed, expected)

    def test_images_factor_is_075_bg_branch_is_none(self):
        aug = iaa.BlendAlphaElementwise(0.75, iaa.Add(10), None)
        observed = aug.augment_image(self.image + 10)
        expected = np.round(
            self.image + 0.75 * (10 + 10) + 0.25 * 10
        ).astype(np.uint8)
        assert np.allclose(observed, expected)

    def test_images_factor_is_tuple(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        aug = iaa.BlendAlphaElementwise((0.0, 1.0), iaa.Add(10), iaa.Add(110))
        observed = (aug.augment_image(image) - 10) / 100
        nb_bins = 10
        hist, _ = np.histogram(
            observed.flatten(), bins=nb_bins, range=(0.0, 1.0), density=False)
        density_expected = 1.0/nb_bins
        density_tolerance = 0.05
        for nb_samples in hist:
            density = nb_samples / observed.size
            assert np.isclose(density, density_expected,
                              rtol=0, atol=density_tolerance)

    def test_bad_datatype_for_factor_fails(self):
        got_exception = False
        try:
            _ = iaa.BlendAlphaElementwise(False, iaa.Add(10), None)
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_images_with_per_channel_in_alpha_and_tuple_as_factor(self):
        image = np.zeros((1, 1, 100), dtype=np.uint8)
        aug = iaa.BlendAlphaElementwise(
            (0.0, 1.0),
            iaa.Add(10),
            iaa.Add(110),
            per_channel=True)
        observed = aug.augment_image(image)
        assert len(set(observed.flatten())) > 1

    def test_bad_datatype_for_per_channel_fails(self):
        got_exception = False
        try:
            _ = iaa.BlendAlphaElementwise(
                0.5,
                iaa.Add(10),
                None,
                per_channel="test")
        except Exception as exc:
            assert "Expected " in str(exc)
            got_exception = True
        assert got_exception

    def test_hooks_limiting_propagation(self):
        aug = iaa.BlendAlphaElementwise(
            0.5,
            iaa.Add(100),
            iaa.Add(50),
            name="AlphaElementwiseTest")

        def propagator(images, augmenter, parents, default):
            if "AlphaElementwise" in augmenter.name:
                return False
            else:
                return default

        hooks = ia.HooksImages(propagator=propagator)
        image = np.zeros((10, 10, 3), dtype=np.uint8) + 1
        observed = aug.augment_image(image, hooks=hooks)
        assert np.array_equal(observed, image)

    def test_heatmaps_and_per_channel_factor_is_zeros(self):
        aug = iaa.BlendAlphaElementwise(
            _DummyMaskParameter(inverted=False),
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"x": -1}),
            per_channel=True)
        observed = aug.augment_heatmaps([self.heatmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps_r1.get_arr())

    def test_heatmaps_and_per_channel_factor_is_ones(self):
        aug = iaa.BlendAlphaElementwise(
            _DummyMaskParameter(inverted=True),
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"x": -1}),
            per_channel=True)
        observed = aug.augment_heatmaps([self.heatmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), self.heatmaps_l1.get_arr())

    def test_segmaps_and_per_channel_factor_is_zeros(self):
        aug = iaa.BlendAlphaElementwise(
            _DummyMaskParameter(inverted=False),
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"x": -1}),
            per_channel=True)
        observed = aug.augment_segmentation_maps([self.segmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert np.array_equal(observed.get_arr(), self.segmaps_r1.get_arr())

    def test_segmaps_and_per_channel_factor_is_ones(self):
        aug = iaa.BlendAlphaElementwise(
            _DummyMaskParameter(inverted=True),
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"x": -1}),
            per_channel=True)
        observed = aug.augment_segmentation_maps([self.segmaps])[0]
        assert observed.shape == (3, 3, 3)
        assert np.array_equal(observed.get_arr(), self.segmaps_l1.get_arr())

    def test_keypoints_factor_is_1(self):
        self._test_cba_factor_is_1("augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_0501(self):
        self._test_cba_factor_is_0501("augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_0(self):
        self._test_cba_factor_is_0("augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_0499(self):
        self._test_cba_factor_is_0499("augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_1_with_per_channel(self):
        self._test_cba_factor_is_1_and_per_channel(
            "augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_0_with_per_channel(self):
        self._test_cba_factor_is_0_and_per_channel(
            "augment_keypoints", self.kpsoi)

    def test_keypoints_factor_is_choice_of_vals_close_050_per_channel(self):
        # TODO can this somehow be integrated into the CBA functions below?
        aug = iaa.BlendAlpha(
            iap.Choice([0.49, 0.51]),
            iaa.Identity(),
            iaa.Affine(translate_px={"x": 1}),
            per_channel=True)
        kpsoi = self.kpsoi

        expected_same = kpsoi.deepcopy()
        expected_both_shifted = kpsoi.shift(x=1)
        expected_fg_shifted = ia.KeypointsOnImage(
            [kpsoi.keypoints[0].shift(x=1), kpsoi.keypoints[1]],
            shape=self.kpsoi.shape)
        expected_bg_shifted = ia.KeypointsOnImage(
            [kpsoi.keypoints[0], kpsoi.keypoints[1].shift(x=1)],
            shape=self.kpsoi.shape)

        seen = [0, 0]
        for _ in sm.xrange(200):
            observed = aug.augment_keypoints([kpsoi])[0]
            if keypoints_equal([observed], [expected_same]):
                seen[0] += 1
            elif keypoints_equal([observed], [expected_both_shifted]):
                seen[1] += 1
            elif keypoints_equal([observed], [expected_fg_shifted]):
                seen[2] += 1
            elif keypoints_equal([observed], [expected_bg_shifted]):
                seen[3] += 1
            else:
                assert False
        assert 100 - 50 < seen[0] < 100 + 50
        assert 100 - 50 < seen[1] < 100 + 50

    def test_keypoints_are_empty(self):
        kpsoi = ia.KeypointsOnImage([], shape=(1, 2, 3))
        self._test_empty_cba("augment_keypoints", kpsoi)

    def test_keypoints_hooks_limit_propagation(self):
        self._test_cba_hooks_limit_propagation("augment_keypoints", self.kpsoi)

    def test_polygons_factor_is_1(self):
        self._test_cba_factor_is_1("augment_polygons", self.psoi)

    def test_polygons_factor_is_0501(self):
        self._test_cba_factor_is_0501("augment_polygons", self.psoi)

    def test_polygons_factor_is_0(self):
        self._test_cba_factor_is_0("augment_polygons", self.psoi)

    def test_polygons_factor_is_0499(self):
        self._test_cba_factor_is_0499("augment_polygons", self.psoi)

    def test_polygons_factor_is_1_and_per_channel(self):
        self._test_cba_factor_is_1_and_per_channel(
            "augment_polygons", self.psoi)

    def test_polygons_factor_is_0_and_per_channel(self):
        self._test_cba_factor_is_0_and_per_channel(
            "augment_polygons", self.psoi)

    def test_polygons_factor_is_choice_around_050_and_per_channel(self):
        # We use more points here to verify the
        # either-or-mode (pointwise=False). The probability that all points
        # move in the same way be coincidence is extremely low for so many.
        ps = [ia.Polygon([(0, 0), (15, 0), (10, 0), (10, 5), (10, 10),
                          (5, 10), (5, 5), (0, 10), (0, 5), (0, 0)])]
        psoi = ia.PolygonsOnImage(ps, shape=(15, 15, 3))
        self._test_cba_factor_is_choice_around_050_and_per_channel(
            "augment_polygons", psoi, pointwise=False
        )

    def test_empty_polygons(self):
        psoi = ia.PolygonsOnImage([], shape=(1, 2, 3))
        self._test_empty_cba("augment_polygons", psoi)

    def test_polygons_hooks_limit_propagation(self):
        self._test_cba_hooks_limit_propagation("augment_polygons", self.psoi)

    def test_line_strings_factor_is_1(self):
        self._test_cba_factor_is_1("augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_0501(self):
        self._test_cba_factor_is_0501("augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_0(self):
        self._test_cba_factor_is_0("augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_0499(self):
        self._test_cba_factor_is_0499("augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_1_and_per_channel(self):
        self._test_cba_factor_is_1_and_per_channel(
            "augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_0_and_per_channel(self):
        self._test_cba_factor_is_0_and_per_channel(
            "augment_line_strings", self.lsoi)

    def test_line_strings_factor_is_choice_around_050_and_per_channel(self):
        # see same polygons test for why self.lsoi is not used here
        lss = [ia.LineString([(0, 0), (15, 0), (10, 0), (10, 5), (10, 10),
                              (5, 10), (5, 5), (0, 10), (0, 5), (0, 0)])]
        lsoi = ia.LineStringsOnImage(lss, shape=(15, 15, 3))
        self._test_cba_factor_is_choice_around_050_and_per_channel(
            "augment_line_strings", lsoi, pointwise=False
        )

    def test_empty_line_strings(self):
        lsoi = ia.LineStringsOnImage([], shape=(1, 2, 3))
        self._test_empty_cba("augment_line_strings", lsoi)

    def test_line_strings_hooks_limit_propagation(self):
        self._test_cba_hooks_limit_propagation(
            "augment_line_strings", self.lsoi)

    def test_bounding_boxes_factor_is_1(self):
        self._test_cba_factor_is_1("augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_0501(self):
        self._test_cba_factor_is_0501("augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_0(self):
        self._test_cba_factor_is_0("augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_0499(self):
        self._test_cba_factor_is_0499("augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_1_and_per_channel(self):
        self._test_cba_factor_is_1_and_per_channel(
            "augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_0_and_per_channel(self):
        self._test_cba_factor_is_0_and_per_channel(
            "augment_bounding_boxes", self.bbsoi)

    def test_bounding_boxes_factor_is_choice_around_050_and_per_channel(self):
        # TODO pointwise=True or False makes no difference here, because
        #      there aren't enough points (see corresponding polygon test)
        self._test_cba_factor_is_choice_around_050_and_per_channel(
            "augment_bounding_boxes", self.bbsoi, pointwise=False
        )

    def test_empty_bounding_boxes(self):
        bbsoi = ia.BoundingBoxesOnImage([], shape=(1, 2, 3))
        self._test_empty_cba("augment_bounding_boxes", bbsoi)

    def test_bounding_boxes_hooks_limit_propagation(self):
        self._test_cba_hooks_limit_propagation(
            "augment_bounding_boxes", self.bbsoi)

    @classmethod
    def _test_cba_factor_is_1(cls, augf_name, cbaoi):
        aug = iaa.BlendAlphaElementwise(
            1.0,
            iaa.Identity(),
            iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)([cbaoi])

        assert_cbaois_equal(observed[0], cbaoi)

    @classmethod
    def _test_cba_factor_is_0501(cls, augf_name, cbaoi):
        aug = iaa.BlendAlphaElementwise(
            0.501,
            iaa.Identity(),
            iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)([cbaoi])

        assert_cbaois_equal(observed[0], cbaoi)

    @classmethod
    def _test_cba_factor_is_0(cls, augf_name, cbaoi):
        aug = iaa.BlendAlphaElementwise(
            0.0,
            iaa.Identity(),
            iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)([cbaoi])

        expected = shift_cbaoi(cbaoi, left=1)
        assert_cbaois_equal(observed[0], expected)

    @classmethod
    def _test_cba_factor_is_0499(cls, augf_name, cbaoi):
        aug = iaa.BlendAlphaElementwise(
            0.499,
            iaa.Identity(),
            iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)([cbaoi])

        expected = shift_cbaoi(cbaoi, left=1)
        assert_cbaois_equal(observed[0], expected)

    @classmethod
    def _test_cba_factor_is_1_and_per_channel(cls, augf_name, cbaoi):
        aug = iaa.BlendAlphaElementwise(
            1.0,
            iaa.Identity(),
            iaa.Affine(translate_px={"x": 1}),
            per_channel=True)

        observed = getattr(aug, augf_name)([cbaoi])

        assert_cbaois_equal(observed[0], cbaoi)

    @classmethod
    def _test_cba_factor_is_0_and_per_channel(cls, augf_name, cbaoi):
        aug = iaa.BlendAlphaElementwise(
            0.0,
            iaa.Identity(),
            iaa.Affine(translate_px={"x": 1}),
            per_channel=True)

        observed = getattr(aug, augf_name)([cbaoi])

        expected = shift_cbaoi(cbaoi, left=1)
        assert_cbaois_equal(observed[0], expected)

    @classmethod
    def _test_cba_factor_is_choice_around_050_and_per_channel(
            cls, augf_name, cbaoi, pointwise):
        aug = iaa.BlendAlphaElementwise(
            iap.Choice([0.49, 0.51]),
            iaa.Identity(),
            iaa.Affine(translate_px={"x": 1}),
            per_channel=True)

        expected_same = cbaoi.deepcopy()
        expected_shifted = shift_cbaoi(cbaoi, left=1)

        nb_iterations = 400
        seen = [0, 0, 0]
        for _ in sm.xrange(nb_iterations):
            observed = getattr(aug, augf_name)([cbaoi])[0]
            # We use here allclose() instead of coords_almost_equals()
            # as the latter one is much slower for polygons and we don't have
            # to deal with tricky geometry changes here, just naive shifting.
            if np.allclose(observed.items[0].coords,
                           expected_same.items[0].coords,
                           rtol=0, atol=0.1):
                seen[0] += 1
            elif np.allclose(observed.items[0].coords,
                             expected_shifted.items[0].coords,
                             rtol=0, atol=0.1):
                seen[1] += 1
            else:
                seen[2] += 1

        if pointwise:
            # This code can be used if the polygon augmentation mode is
            # AlphaElementwise._MODE_POINTWISE. Currently it is _MODE_EITHER_OR.
            nb_points = len(cbaoi.items[0].coords)
            p_all_same = 2 * ((1/2)**nb_points)  # all points moved in same way
            expected_iter = nb_iterations*p_all_same
            expected_iter_notsame = nb_iterations*(1-p_all_same)
            atol = nb_iterations * (5*p_all_same)

            assert np.isclose(seen[0], expected_iter, rtol=0, atol=atol)
            assert np.isclose(seen[1], expected_iter, rtol=0, atol=atol)
            assert np.isclose(seen[2], expected_iter_notsame, rtol=0, atol=atol)
        else:
            expected_iter = nb_iterations*0.5
            atol = nb_iterations*0.15
            assert np.isclose(seen[0], expected_iter, rtol=0, atol=atol)
            assert np.isclose(seen[1], expected_iter, rtol=0, atol=atol)
            assert seen[2] == 0

    @classmethod
    def _test_empty_cba(cls, augf_name, cbaoi):
        aug = iaa.BlendAlphaElementwise(
            0.501,
            iaa.Identity(),
            iaa.Affine(translate_px={"x": 1}))

        observed = getattr(aug, augf_name)(cbaoi)

        assert len(observed.items) == 0
        assert observed.shape == (1, 2, 3)

    @classmethod
    def _test_cba_hooks_limit_propagation(cls, augf_name, cbaoi):
        aug = iaa.BlendAlphaElementwise(
            0.0,
            iaa.Affine(translate_px={"x": 1}),
            iaa.Affine(translate_px={"y": 1}),
            name="AlphaTest")

        def propagator(cbaoi_to_aug, augmenter, parents, default):
            if "Alpha" in augmenter.name:
                return False
            else:
                return default

        # no hooks for polygons yet, so we use HooksKeypoints
        hooks = ia.HooksKeypoints(propagator=propagator)
        observed = getattr(aug, augf_name)([cbaoi], hooks=hooks)[0]
        assert observed.items[0].coords_almost_equals(cbaoi.items[0])

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 0, dtype=np.uint8)
                aug = iaa.BlendAlpha(1.0, iaa.Add(1), iaa.Add(100))

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [
            (1, 1, 4),
            (1, 1, 5),
            (1, 1, 512),
            (1, 1, 513)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 0, dtype=np.uint8)
                aug = iaa.BlendAlpha(1.0, iaa.Add(1), iaa.Add(100))

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_pickleable(self):
        aug = iaa.BlendAlphaElementwise(
            (0.1, 0.9),
            iaa.Add((1, 10), random_state=1),
            iaa.Add((11, 20), random_state=2),
            per_channel=True,
            random_state=3)
        runtest_pickleable_uint8_img(aug, iterations=3)


class TestBlendAlphaSomeColors(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        child1 = iaa.Sequential([])
        child2 = iaa.Sequential([])
        aug = iaa.BlendAlphaSomeColors(child1, child2)
        assert aug.foreground is child1
        assert aug.background is child2
        assert isinstance(aug.mask_generator, iaa.SomeColorsMaskGen)

    def test_grayscale_drops_different_colors(self):
        image = np.uint8([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 128],
            [128, 255, 128],
            [128, 128, 255]
        ]).reshape((1, 9, 3))
        image_gray = iaa.Grayscale(1.0)(image=image)
        aug = iaa.BlendAlphaSomeColors(iaa.Grayscale(1.0),
                                       nb_bins=256, smoothness=0)

        nb_grayscaled = []
        for _ in sm.xrange(50):
            image_aug = aug(image=image)
            grayscaled = np.sum((image_aug == image_gray).astype(np.int32),
                                axis=2)
            assert np.all(np.logical_or(grayscaled == 0, grayscaled == 3))
            nb_grayscaled.append(np.sum(grayscaled == 3))

        assert len(set(nb_grayscaled)) >= 5

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0, 3),
            (0, 1, 3),
            (1, 0, 3)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.full(shape, 0, dtype=np.uint8)
                aug = iaa.BlendAlphaSomeColors(iaa.Add(1), iaa.Add(100))

                image_aug = aug(image=image)

                assert np.all(image_aug == 1)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_pickleable(self):
        aug = iaa.BlendAlphaSomeColors(
            iaa.Add((1, 10), random_state=1),
            iaa.Add((11, 20), random_state=2),
            random_state=3)
        runtest_pickleable_uint8_img(aug, iterations=3)


class TestStochasticParameterMaskGen(unittest.TestCase):
    def _test_draw_masks_nhwc(self, shape):
        batch = ia.BatchInAugmentation(
            images=np.zeros(shape, dtype=np.uint8)
        )
        values = np.float32([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        param = iap.DeterministicList(values.flatten())

        gen = iaa.StochasticParameterMaskGen(param, per_channel=False)

        masks = gen.draw_masks(batch, random_state=0)

        for i in np.arange(shape[0]):
            assert np.allclose(masks[i], values)

    def test_draw_masks_hw3_images(self):
        self._test_draw_masks_nhwc((2, 2, 3, 3))

    def test_draw_masks_hw1_images(self):
        self._test_draw_masks_nhwc((2, 2, 3, 1))

    def test_draw_masks_hw_images(self):
        self._test_draw_masks_nhwc((2, 2, 3))

    def test_draw_masks_batch_without_images(self):
        bb = ia.BoundingBox(x1=1, y1=2, x2=3, y2=4)
        bbsoi1 = ia.BoundingBoxesOnImage([bb], shape=(2, 3, 3))
        bbsoi2 = ia.BoundingBoxesOnImage([], shape=(3, 3, 3))
        batch = ia.BatchInAugmentation(
            bounding_boxes=[bbsoi1, bbsoi2]
        )
        # sampling for shape of bbsoi1 will cover row1 and row2, then
        # sampling for bbsoi2 will cover row1, row2, row3
        # masks are sampled independently per row/image, so it starts over
        # again for bbsoi2
        values = np.float32([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        param = iap.DeterministicList(values.flatten())

        gen = iaa.StochasticParameterMaskGen(param, per_channel=False)

        masks = gen.draw_masks(batch, random_state=0)

        expected1 = values[0:2]
        expected2 = values[0:3]
        assert np.allclose(masks[0], expected1)
        assert np.allclose(masks[1], expected2)

    def test_per_channel(self):
        for per_channel in [True, iap.Deterministic(0.51)]:
            batch = ia.BatchInAugmentation(
                images=np.zeros((1, 2, 3, 2), dtype=np.uint8)
            )
            values = np.float32([
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.10, 0.11, 0.12]
            ])
            param = iap.DeterministicList(values.flatten())

            gen = iaa.StochasticParameterMaskGen(param,
                                                 per_channel=per_channel)

            masks = gen.draw_masks(batch, random_state=0)

            assert np.allclose(masks[0], values.reshape((2, 3, 2)))

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for per_channel in [False, True]:
            for shape in shapes:
                with self.subTest(per_channel=per_channel, shape=shape):
                    image = np.zeros(shape, dtype=np.uint8)
                    batch = ia.BatchInAugmentation(
                        images=[np.zeros(shape, dtype=np.uint8)]
                    )
                    param = iap.Deterministic(1.0)
                    gen = iaa.StochasticParameterMaskGen(
                        param, per_channel=per_channel)

                    masks = gen.draw_masks(batch, random_state=0)

                    assert len(masks) == 1
                    if not per_channel:
                        assert masks[0].shape == shape[0:2]
                    else:
                        assert masks[0].shape == shape


class TestSomeColorsMaskGen(unittest.TestCase):
    def test___init___defaults(self):
        gen = iaa.SomeColorsMaskGen()
        assert np.isclose(gen.nb_bins.a.value, 5)
        assert np.isclose(gen.nb_bins.b.value, 15)
        assert np.isclose(gen.smoothness.a.value, 0.1)
        assert np.isclose(gen.smoothness.b.value, 0.3)
        assert np.isclose(gen.alpha.a[0], 0.0)
        assert np.isclose(gen.alpha.a[1], 1.0)
        assert np.isclose(gen.rotation_deg.a.value, 0)
        assert np.isclose(gen.rotation_deg.b.value, 360)
        assert gen.from_colorspace == iaa.CSPACE_RGB

    def test___init___custom_settings(self):
        gen = iaa.SomeColorsMaskGen(
            nb_bins=100,
            smoothness=0.5,
            alpha=0.7,
            rotation_deg=123,
            from_colorspace=iaa.CSPACE_HSV
        )
        assert gen.nb_bins.value == 100
        assert np.isclose(gen.smoothness.value, 0.5)
        assert np.isclose(gen.alpha.value, 0.7)
        assert np.isclose(gen.rotation_deg.value, 123)
        assert gen.from_colorspace == iaa.CSPACE_HSV

    def test_draw_masks_marks_different_colors(self):
        image = np.uint8([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 128],
            [128, 255, 128],
            [128, 128, 255]
        ]).reshape((9, 1, 3))
        image = np.tile(image, (9, 50, 1))
        batch = ia.BatchInAugmentation(images=[image])
        gen = iaa.SomeColorsMaskGen(nb_bins=256, smoothness=0,
                                    alpha=[0, 1])
        expected_mask_sums = np.arange(1 + image.shape[0]) * image.shape[1]
        expected_mask_sums = expected_mask_sums.astype(np.float32)

        mask_sums = []
        for i in sm.xrange(50):
            mask = gen.draw_masks(batch, random_state=i)[0]

            mask_sum = int(np.sum(mask))
            mask_sums.append(mask_sum)

            assert np.any(
                np.isclose(
                    np.min(np.abs(expected_mask_sums - mask_sum)),
                    0.0,
                    rtol=0,
                    atol=0.01)
            )
            assert mask.shape == image.shape[0:2]
            assert mask.dtype.name == "float32"

        assert len(np.unique(mask_sums)) >= 4

    def test_draw_masks_marks_alpha_is_0(self):
        image = np.uint8([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 128],
            [128, 255, 128],
            [128, 128, 255]
        ]).reshape((1, 9, 3))
        batch = ia.BatchInAugmentation(images=[image])
        gen = iaa.SomeColorsMaskGen(alpha=0.0)

        mask = gen.draw_masks(batch)[0]

        assert np.allclose(mask, 0.0)

    def test_draw_masks_alpha_is_1(self):
        image = np.uint8([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 128],
            [128, 255, 128],
            [128, 128, 255]
        ]).reshape((1, 9, 3))
        batch = ia.BatchInAugmentation(images=[image])
        gen = iaa.SomeColorsMaskGen(alpha=1.0)

        mask = gen.draw_masks(batch)[0]

        assert np.allclose(mask, 1.0)

    @mock.patch("imgaug.augmenters.color.change_colorspace_")
    def test_from_colorspace(self, mock_cc):
        image = np.uint8([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 128],
            [128, 255, 128],
            [128, 128, 255]
        ]).reshape((1, 9, 3))
        batch = ia.BatchInAugmentation(images=[image])
        mock_cc.return_value = np.copy(image)
        gen = iaa.SomeColorsMaskGen(alpha=1.0, from_colorspace=iaa.CSPACE_BGR)

        _ = gen.draw_masks(batch)

        assert mock_cc.call_count == 1
        assert np.array_equal(mock_cc.call_args_list[0][0][0], image)
        assert (mock_cc.call_args_list[0][1]["to_colorspace"]
                == iaa.CSPACE_HSV)
        assert (mock_cc.call_args_list[0][1]["from_colorspace"]
                == iaa.CSPACE_BGR)

    def test__upscale_to_256_alpha_bins__1_to_256(self):
        alphas = np.float32([0.5])

        alphas_up = iaa.SomeColorsMaskGen._upscale_to_256_alpha_bins(alphas)

        assert alphas_up.shape == (256,)
        assert np.allclose(alphas_up, 0.5)

    def test__upscale_to_256_alpha_bins__2_to_256(self):
        alphas = np.float32([1.0, 0.5])

        alphas_up = iaa.SomeColorsMaskGen._upscale_to_256_alpha_bins(alphas)

        assert alphas_up.shape == (256,)
        assert np.allclose(alphas_up[0:128], 1.0)
        assert np.allclose(alphas_up[128:], 0.5)

    def test__upscale_to_256_alpha_bins__255_to_256(self):
        alphas = np.zeros((255,), dtype=np.float32)
        alphas[0] = 0.25
        alphas[1:254] = 0.5
        alphas[254] = 1.0

        alphas_up = iaa.SomeColorsMaskGen._upscale_to_256_alpha_bins(alphas)

        assert alphas_up.shape == (256,)
        assert np.allclose(alphas_up[0:2], 0.25)
        assert np.allclose(alphas_up[2:], 0.5)

    def test__upscale_to_256_alpha_bins__256_to_256(self):
        alphas = np.full((256,), 0.5, dtype=np.float32)

        alphas_up = iaa.SomeColorsMaskGen._upscale_to_256_alpha_bins(alphas)

        assert alphas_up.shape == (256,)
        assert np.allclose(alphas, 0.5)

    def test__rotate_alpha_bins__by_0(self):
        alphas = np.linspace(0.0, 1.0, 256)

        alphas_rot = iaa.SomeColorsMaskGen._rotate_alpha_bins(alphas, 0)

        assert np.allclose(alphas_rot, alphas)

    def test__rotate_alpha_bins__by_1(self):
        alphas = np.linspace(0.0, 1.0, 256)

        alphas_rot = iaa.SomeColorsMaskGen._rotate_alpha_bins(alphas, 1)

        assert np.allclose(alphas_rot[:-1], alphas[1:])
        assert np.allclose(alphas_rot[-1:], alphas[:1])

    def test__rotate_alpha_bins__by_255(self):
        alphas = np.linspace(0.0, 1.0, 256)

        alphas_rot = iaa.SomeColorsMaskGen._rotate_alpha_bins(alphas, 255)

        assert np.allclose(alphas_rot[:-255], alphas[255:])
        assert np.allclose(alphas_rot[-255:], alphas[:255])

    def test__rotate_alpha_bins__by_256(self):
        alphas = np.linspace(0.0, 1.0, 256)

        alphas_rot = iaa.SomeColorsMaskGen._rotate_alpha_bins(alphas, 256)

        assert np.allclose(alphas_rot, alphas)

    def test__smoothen_alphas__0(self):
        alphas = np.zeros((11,), dtype=np.float32)
        alphas[5-3:5+3+1] = 1.0

        alphas_smooth = iaa.SomeColorsMaskGen._smoothen_alphas(alphas, 0.0)

        assert np.allclose(alphas_smooth, alphas)

    def test__smoothen_alphas__002(self):
        alphas = np.zeros((11,), dtype=np.float32)
        alphas[5-3:5+3+1] = 1.0

        alphas_smooth = iaa.SomeColorsMaskGen._smoothen_alphas(alphas, 0.02)

        assert np.allclose(alphas_smooth, alphas, atol=0.02)

    def test__smoothen_alphas__1(self):
        alphas = np.zeros((11,), dtype=np.float32)
        alphas[5-3:5+3+1] = 1.0

        alphas_smooth = iaa.SomeColorsMaskGen._smoothen_alphas(alphas, 1.0)

        assert np.isclose(alphas_smooth[0], 0.0, atol=0.01)
        assert not np.isclose(alphas_smooth[2], 1.0, atol=0.1)
        assert np.isclose(alphas_smooth[5], 1.0, atol=0.01)

    def test__generate_pixelwise_alpha_map(self):
        image_hsv = np.uint8([
            [0, 0, 0],
            [50, 0, 0],
            [100, 0, 0],
            [150, 0, 0],
            [200, 0, 0],
            [250, 0, 0],
            [255, 0, 0]
        ]).reshape((1, 7, 3))
        hue_to_alpha = np.zeros((256,), dtype=np.float32)
        hue_to_alpha[0] = 0.1
        hue_to_alpha[50] = 0.2
        hue_to_alpha[100] = 0.3
        hue_to_alpha[150] = 0.4
        hue_to_alpha[200] = 0.5
        hue_to_alpha[250] = 0.6
        hue_to_alpha[255] = 0.7

        mask = iaa.SomeColorsMaskGen._generate_pixelwise_alpha_mask(
            image_hsv, hue_to_alpha)

        # a bit of tolerance here due to the mask being converted from
        # [0, 255] to [0.0, 1.0]
        assert np.allclose(
            mask.flatten(),
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            atol=0.05)

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0, 3),
            (0, 1, 3),
            (1, 0, 3)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                batch = ia.BatchInAugmentation(images=[image])
                gen = iaa.SomeColorsMaskGen()

                mask = gen.draw_masks(batch)[0]

                assert mask.shape == shape[0:2]
                assert mask.dtype.name == "float32"

    def test_batch_contains_no_images(self):
        hms = ia.HeatmapsOnImage(np.zeros((5, 5), dtype=np.float32),
                                 shape=(10, 10, 3))
        batch = ia.BatchInAugmentation(heatmaps=[hms])
        gen = iaa.SomeColorsMaskGen()

        with self.assertRaises(AssertionError):
            _masks = gen.draw_masks(batch)


class TestSimplexNoiseAlpha(unittest.TestCase):
    def test_deprecation_warning(self):
        aug1 = iaa.Sequential([])
        aug2 = iaa.Sequential([])

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            aug = iaa.SimplexNoiseAlpha(first=aug1, second=aug2)

            assert (
                "is deprecated"
                in str(caught_warnings[-1].message)
            )

        assert isinstance(aug, iaa.BlendAlphaSimplexNoise)
        assert aug.foreground is aug1
        assert aug.background is aug2


class TestFrequencyNoiseAlpha(unittest.TestCase):
    def test_deprecation_warning(self):
        aug1 = iaa.Sequential([])
        aug2 = iaa.Sequential([])

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            aug = iaa.FrequencyNoiseAlpha(first=aug1, second=aug2)

            assert (
                "is deprecated"
                in str(caught_warnings[-1].message)
            )

        assert isinstance(aug, iaa.BlendAlphaFrequencyNoise)
        assert aug.foreground is aug1
        assert aug.background is aug2
