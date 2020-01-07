# Improved Blending #???

* Added `imgaug.augmenters.blend.BlendAlphaMask`, which uses a mask generator
  instance to generate per batch alpha masks and then alpha-blends using
  these masks.
* Added `imgaug.augmenters.blend.IBatchwiseMaskGenerator`, an interface for
  classes generating masks on a batch-by-batch basis.
* Added `imgaug.augmenters.blend.StochasticParameterMaskGen`, a mask generator
  helper for `BlendAlphaMask`.
* Changed `imgaug.parameters.SimplexNoise` and
  `imgaug.parameters.FrequencyNoise` to also accept `(H, W, C)` sampling
  shapes, instead of only `(H, W)`.
* Refactored `AlphaElementwise` to be a wrapper around `BlendAlphaMask`.
* Renamed `AlphaElementwise` to `BlendAlphaElementwise`.
  `AlphaElementwise` is now deprecated.
* Renamed `SimplexNoiseAlpha` to `BlendAlphaSimplexNoise`.
  `SimplexNoiseAlpha` is now deprecated.
* Renamed `FrequencyNoiseAlpha` to `BlendAlphaFrequencyNoise`.
  `FrequencyNoiseAlpha` is now deprecated.
* Renamed arguments `first` and `second` to `foreground` and `background`
  in `BlendAlphaElementwise`, `SimplexNoiseAlpha` and `FrequencyNoiseAlpha`.
