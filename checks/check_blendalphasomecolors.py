from __future__ import print_function, division, absolute_import
import imageio
import imgaug as ia
import imgaug.augmenters as iaa


def main():
    aug = iaa.BlendAlphaMask(
        iaa.SomeColorsMaskGen(),
        iaa.OneOf([
            iaa.TotalDropout(1.0),
            iaa.AveragePooling(8)
        ])
    )

    aug2 = iaa.BlendAlphaSomeColors(iaa.OneOf([
            iaa.TotalDropout(1.0),
            iaa.AveragePooling(8)
    ]))

    url = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/"
        "Vincent_van_Gogh_-_Wheatfield_with_crows_-_Google_Art_Project.jpg/"
        "320px-Vincent_van_Gogh_-_Wheatfield_with_crows_-_Google_Art_Project"
        ".jpg")
    img = imageio.imread(url)

    ia.imshow(ia.draw_grid(aug(images=[img]*25), cols=5, rows=5))
    ia.imshow(ia.draw_grid(aug2(images=[img]*25), cols=5, rows=5))

if __name__ == "__main__":
    main()
