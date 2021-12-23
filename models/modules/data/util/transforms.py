import random
from typing import Callable, Dict, Tuple

import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
import torchvision.transforms.functional as F
from PIL.Image import Image

import cv2
import pathlib

FILLCOLOR = (255, 255, 255)


def transforms_info() -> Dict[
    str, Tuple[Callable[[Image, float], Image], float, float]
]:
    """Return augmentation functions and their ranges."""
    transforms_list = [
        (Identity, 0.0, 0.0),
        (Invert, 0.0, 0.0),
        (Contrast, 0.0, 0.9),
        (AutoContrast, 0.0, 0.0),
        # (Rotate, 0.0, 30.0),
        (TranslateX, 0.0, 0.2),
        (TranslateX, 0.0, 0.2),
        (Sharpness, 0.0, 0.9),
        (ShearX, 0.0, 0.3),
        (ShearY, 0.0, 0.3),
        (Color, 0.0, 0.9),
        (Brightness, 0.0, 0.9),
        (Equalize, 0.0, 0.0),
        (Solarize, 256.0, 0.0),
        (Posterize, 8, 4)
    ]
    return {f.__name__: (f, low, high) for f, low, high in transforms_list}


def Identity(img: Image, bboxes:list, transcripts:list, _: float) -> Image:
    """Identity map."""
    return img, bboxes, transcripts


def Invert(img: Image, bboxes:list, transcripts:list, _: float) -> Image: # 색반전
    """Invert the image."""
    return PIL.ImageOps.invert(img), bboxes, transcripts


def Contrast(img: Image, bboxes:list, transcripts:list, magnitude: float) -> Image: #대비강조
    """Put contrast effect on the image."""
    return PIL.ImageEnhance.Contrast(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    ), bboxes, transcripts


def AutoContrast(img: Image, bboxes:list, transcripts:list, _: float) -> Image:
    """Put contrast effect on the image."""
    return PIL.ImageOps.autocontrast(img), bboxes, transcripts

def TranslateX(img: Image, bboxes:list, transcripts:list, magnitude: float) -> Image:
    """Translate the image on x-axis."""
    level = magnitude * img.size[0] * random.choice([-1, 1])
    image = img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, 0, level, 0, 1, 0),
        fillcolor=FILLCOLOR,
    )

    # ver1
    width, _ = img.size
    for idx, bbox in enumerate(bboxes):
        drop_bbox = False
        for idx in range(4):
            bbox[idx][0] = bbox[idx][0] - level
            if bbox[idx][0] < 0 or bbox[idx][0] > width:
                transcripts[idx] = "*"
    return image, bboxes, transcripts

    # ver2
    # width, _ = img.size
    # trans_bboxes = []
    # trans_transcripts = []
    # for idx, bbox in enumerate(bboxes):
    #     drop_bbox = False
    #     for idx in range(4):
    #         bbox[idx][0] = bbox[idx][0] - level
    #         if bbox[idx][0] < 0 or bbox[idx][0] > width:
    #             drop_bbox = True
    #             break
    #     if drop_bbox == False:
    #         trans_bboxes.append(bbox)
    #         trans_transcripts.append(transcripts[idx])
    # trans_bboxes = np.array(trans_bboxes)
    # return image, trans_bboxes, trans_transcripts


def TranslateY(img: Image, bboxes:list, transcripts:list, magnitude: float) -> Image:
    """Translate the image on y-axis."""
    level = magnitude * img.size[0] * random.choice([-1, 1])
    image = img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, 0, 0, 0, 1, level),
        fillcolor=FILLCOLOR,
    )

    _, height = img.size
    for bbox in bboxes:
        drop_bbox = False
        for idx in range(4):
            bbox[idx][1] = bbox[idx][1] - level
            if bbox[idx][1] < 0 or bbox[idx][1] > height:
                transcripts[idx] = '*'
    return image, bboxes, transcripts

    # _, height = img.size
    # trans_bboxes = []
    # trans_transcripts = []
    # for bbox in bboxes:
    #     drop_bbox = False
    #     for idx in range(4):
    #         bbox[idx][1] = bbox[idx][1] - level
    #         if bbox[idx][1] < 0 or bbox[idx][1] > height:
    #             drop_bbox = True
    #             break
    #     if drop_bbox == False:
    #         trans_bboxes.append(bbox)
    #         trans_transcripts.append(transcripts[idx])
    # trans_bboxes = np.array(trans_bboxes)
    # return image, trans_bboxes, trans_transcripts



def Sharpness(img: Image, bboxes:list, transcripts:list, magnitude: float) -> Image:
    """Adjust the sharpness of the image."""
    return PIL.ImageEnhance.Sharpness(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    ), bboxes, transcripts


def ShearX(img: Image, bboxes:list, transcripts:list, magnitude: float) -> Image:
    """Shear the image on x-axis."""
    level = magnitude * random.choice([-1, 1])
    image = img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, level, 0, 0, 1, 0),
        PIL.Image.BICUBIC,
        fillcolor=FILLCOLOR,
    )

    # ver1
    width, height = img.size
    for bbox in bboxes:
        drop_bbox = False
        for idx in range(4):
            h_ratio = bbox[idx][1] / height
            bbox[idx][0] = bbox[idx][0] - width * level * h_ratio
            if bbox[idx][0] < 0 or bbox[idx][0] > width:
                transcripts[idx] = '*'
    return image, bboxes, transcripts


    # ver2
    # width, height = img.size
    # trans_bboxes = []
    # trans_transcripts = []
    # for bbox in bboxes:
    #     drop_bbox = False
    #     for idx in range(4):
    #         h_ratio = bbox[idx][1] / height
    #         bbox[idx][0] = bbox[idx][0] - width * level * h_ratio
    #         if bbox[idx][0] < 0 or bbox[idx][0] > width:
    #             drop_bbox = True
    #             break
    #     if drop_bbox == False:
    #         trans_bboxes.append(bbox)
    #         trans_transcripts.append(transcripts[idx])
    # trans_bboxes = np.array(trans_bboxes)
    # return image, trans_bboxes, trans_transcripts


def ShearY(img: Image, bboxes:list, transcripts:list, magnitude: float) -> Image:
    """Shear the image on y-axis."""
    level = magnitude * random.choice([-1, 1])
    image = img.transform(
        img.size,
        PIL.Image.AFFINE,
        (1, 0, 0, level, 1, 0),
        PIL.Image.BICUBIC,
        fillcolor=FILLCOLOR,
    )
    
    width, height = img.size
    for bbox in bboxes:
        drop_bbox = False
        for idx in range(4):
            w_ratio = bbox[idx][0] / width
            bbox[idx][1] = bbox[idx][1] - height * level * w_ratio
            if bbox[idx][1] < 0 or bbox[idx][0] > height:
                transcripts[idx] = '*'
    return image, bboxes, transcripts

    # ver2
    # width, height = img.size
    # trans_bboxes = []
    # trans_transcripts = []
    # for bbox in bboxes:
    #     drop_bbox = False
    #     for idx in range(4):
    #         w_ratio = bbox[idx][0] / width
    #         bbox[idx][1] = bbox[idx][1] - height * level * w_ratio
    #         if bbox[idx][1] < 0 or bbox[idx][0] > height:
    #             drop_bbox = True
    #             break
    #     if drop_bbox == False:
    #         trans_bboxes.append(bbox)
    #         trans_transcripts.append(transcripts[idx])
    
    # trans_bboxes = np.array(trans_bboxes)

    # return image, trans_bboxes, trans_transcripts


def Color(img: Image, bboxes:list, transcripts:list, magnitude: float) -> Image:
    """Adjust the color balance of the image."""
    return PIL.ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])), bboxes, transcripts


def Brightness(img: Image, bboxes:list, transcripts:list, magnitude: float) -> Image:
    """Adjust brightness of the image."""
    return PIL.ImageEnhance.Brightness(img).enhance(
        1 + magnitude * random.choice([-1, 1])
    ), bboxes, transcripts


def Equalize(img: Image, bboxes:list, transcripts:list, _: float) -> Image:
    """Equalize the image."""
    return PIL.ImageOps.equalize(img), bboxes, transcripts


def Solarize(img: Image, bboxes:list, transcripts:list, magnitude: float) -> Image:
    """Solarize the image."""
    return PIL.ImageOps.solarize(img, magnitude), bboxes, transcripts


def Posterize(img: Image, bboxes:list, transcripts:list, magnitude: float) -> Image:
    """Posterize the image."""
    magnitude = int(magnitude)
    return PIL.ImageOps.posterize(img, magnitude), bboxes, transcripts

if __name__=='__main__':
    image_name = 'img_172.jpg'
    image_path = '/opt/ml/project/models/datasets/train_images_resize/'#+image_name
    new_image_path = '/opt/ml/project/models/datasets/transformed/'#+image_name

    image = cv2.imread(image_path+image_name)
    image=PIL.Image.fromarray(np.uint8(image))
    bboxes=[np.array([[0,0],[256,0],[256,256], [0,256]]),
            np.array([[300, 0],[500,0],[500,200], [300,200]]),
            np.array([[0,300],[200,300],[200,500], [0,500]])]
    transform_infos = transforms_info()

    transform_list = list(transform_infos)
    chosen_transforms = random.sample(transform_list, k=1)


    for idx, trans in enumerate(chosen_transforms):
        transform_func, low, high = transform_infos[trans]
        level = random.uniform(low, high)
        image, bboxes = transform_func(image, bboxes, level)
    # print(bboxes)
    image = np.array(image)
    cv2.imwrite(new_image_path+'img_random.png', image)