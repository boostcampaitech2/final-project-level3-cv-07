import time
from fastapi import UploadFile, File, APIRouter
from typing import List
import base64

import datetime
import random
from server.modules.papago import translation_en2ko
from server.modules.generate_point import get_random_polygon, get_random_point
from server.modules.util import read_imagefile

demo_router = APIRouter(prefix='/bbox_demo')

@demo_router.get("/", tags=['demo'])
def read_root():
    return "Boost Camp AI tech CV7's API demo router"


@demo_router.post("/image", tags=['demo'])
async def demo_image(file: UploadFile = File(...)):
    time_start = time.monotonic()
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        print(datetime.datetime.now())
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    print("success image upload!!")
    prediction=[random.randint(0,5)]
    for idx in range(prediction[0]):
        str_trns=translation_en2ko(f"demo trasnslation of bbox{idx}")
        print(f'Papago responese : {str_trns}')
        if str_trns[0]==200:
            prediction.append(
                {
                    'translation':str_trns[1],
                    'point':get_random_polygon()}
                )
        else:
            prediction.append(
                {
                    'translation':f'Papago API Error [{str_trns[0]}]...',
                    'point':[]}
                )
    running_time = time.monotonic() - time_start
    print(datetime.datetime.now())
    print(f'inference time : {running_time:.2f}s')

    return prediction

@demo_router.post("/base64", tags=['demo'])
async def demo_base64(file: UploadFile = File(...)):
    time_start = time.monotonic()
    image = read_imagefile(base64.b64decode(await file.read()))
    print(type(image))
    print("success image upload!!")
    prediction=[random.randint(0,5)]
    for idx in range(prediction[0]):
        str_trns=translation_en2ko(f"demo trasnslation of bbox{idx}")
        print(f'Papago responese : {str_trns}')
        if str_trns[0]==200:
            prediction.append(
                {
                    'translation':str_trns[1],
                    'point':get_random_polygon()}
                )
        else:
            prediction.append(
                {
                    'translation':f'Papago API Error [{str_trns[0]}]...',
                    'point':[]}
                )
    running_time = time.monotonic() - time_start
    print(datetime.datetime.now())
    print(f'inference time : {running_time:.2f}s')

    return prediction

@demo_router.post("/nopapago", tags=['demo'])
async def demo_base64(file: UploadFile = File(...)):
    time_start = time.monotonic()
    image = read_imagefile(base64.b64decode(await file.read()))
    print("success image upload!!")
    prediction=[random.randint(0,5)]
    for idx in range(prediction[0]):
        prediction.append(
            {
                'translation':f"demo trasnslation of bbox{idx}",
                'point':get_random_polygon()}
            )

    running_time = time.monotonic() - time_start
    print(datetime.datetime.now())
    print(f'inference time : {running_time:.2f}s')

    return prediction
