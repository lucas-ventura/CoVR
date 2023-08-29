from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from src.data.randaugment import RandomAugment

normalize = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
)


class transform_train:
    def __init__(self, image_size=384, min_scale=0.5):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                RandomAugment(
                    2,
                    5,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Brightness",
                        "Sharpness",
                        "Equalize",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __call__(self, img):
        return self.transform(img)


class transform_test(transforms.Compose):
    def __init__(self, image_size=384):
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __call__(self, img):
        return self.transform(img)
