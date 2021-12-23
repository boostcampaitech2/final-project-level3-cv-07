from PIL import Image
from torchvision import transforms
from io import BytesIO
import cv2
import numpy as np

trsf=transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ]
    )

def read_imagefile(file) -> Image.Image:
    # image = cv2.imread(file)
    nparr = np.fromstring(file, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # print(f'IMAGE SIZE: {image.shape}')
    # print(type(image))
    # cv2.imwrite('server/image/test.jpg', image)
    # print("success image upload!!")
    return image

def read_imagebase(file) -> cv2:
    # image = Image.open(BytesIO(file))
    image = cv2.imdecode(file, cv2.IMREAD_COLOR)
    # print(f'IMAGE SIZE: {image.shape}')
    # print(type(image))
    # cv2.imwrite('server/image/test.jpg', image)
    print("success image upload!!")
    return image
