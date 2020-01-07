# Improved Blending #???

* Added `imgaug.augmenters.blend.BlendAlphaMask`, which uses a mask generator
  instance to generate per batch alpha masks and then alpha-blends using
  these masks.
* Added `imgaug.augmenters.blend.IBatchwiseMaskGenerator`, an interface for
  classes generating masks on a batch-by-batch basis.
* Added `imgaug.augmenters.blend.StochasticParameterMaskGen`, a mask generator
  helper for `BlendAlphaMask`.
