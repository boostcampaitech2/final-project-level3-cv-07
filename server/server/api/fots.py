import time
from fastapi import UploadFile, File, APIRouter
from typing import List
import base64

import datetime
from server.modules.papago import translation_en2ko
from server.modules.util import read_imagefile,read_imagebase
from server.modules.inference import predict

from server.modules.color_finder import color_list

fots_router = APIRouter(prefix='/fots')

@fots_router.get("/", tags=['fots'])
def read_root():
    return "Boost Camp AI tech CV7's API fots router"



@fots_router.post("/image", tags=['fots'])
async def fots_image(file: UploadFile = File(...)):
    time_start = time.monotonic()
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        print(datetime.datetime.now())
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    return make_fots_response_no_split(image,time_start)

@fots_router.post("/base64", tags=['fots'])
async def fots_base64(file: UploadFile = File(...)):
    time_start = time.monotonic()
    image = read_imagefile(base64.b64decode(await file.read()))

    return make_fots_response_no_split(image,time_start)

@fots_router.post("/image/nopapago", tags=['fots'])
async def fots_image_nopapago(file: UploadFile = File(...)):
    time_start = time.monotonic()
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        print(datetime.datetime.now())
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    print("start predict")
    boxes, pred_transcripts = predict(image)
    print("stop predict")
    prediction=[len(boxes)]
    for idx,(bbox,text) in enumerate(zip(boxes, pred_transcripts)):
        str_trns=(200,text)
        # print(f'Papago responese : {str_trns}')
        if str_trns[0]==200:
            prediction.append(
                {
                    'translation':str_trns[1],
                    'point':bbox.tolist()}
                )
        else:
            prediction.append(
                {
                    'translation':f'Papago API Error [{str_trns[0]}]...',
                    'point':bbox.tolist()}
                )
    running_time = time.monotonic() - time_start
    print(datetime.datetime.now())
    print(f'inference time : {running_time:.2f}s')

    return prediction

@fots_router.post("/base64/nopapago", tags=['fots'])
async def fots_base64_nopapago(file: UploadFile = File(...)):
    time_start = time.monotonic()
    image = read_imagefile(base64.b64decode(await file.read()))
    boxes, pred_transcripts = predict(image)
    prediction=[len(boxes)]
    for idx,(bbox,text) in enumerate(zip(boxes, pred_transcripts)):
        str_trns=(200,text)
        # print(f'Papago responese : {str_trns}')
        if str_trns[0]==200:
            prediction.append(
                {
                    'translation':str_trns[1],
                    'point':bbox.tolist()}
                )
        else:
            prediction.append(
                {
                    'translation':f'Papago API Error [{str_trns[0]}]...',
                    'point':bbox.tolist()}
                )
    running_time = time.monotonic() - time_start
    print(datetime.datetime.now())
    print(f'inference time : {running_time:.2f}s')
    print(prediction)
    return prediction


def make_fots_response(image,time_start):
    boxes, pred_transcripts = predict(image)
    colors = color_list(image, boxes)
    print(f'Detect Text : {pred_transcripts}')
    join_str=" @ "
    print(join_str.join(pred_transcripts))
    resp_code_papago,result_papago = translation_en2ko(join_str.join(pred_transcripts))
    print(result_papago)
    result_papago=result_papago.split(join_str)
    print(f'Translated Text : {result_papago}')
    prediction=[len(boxes)]
    for idx,(bbox,text,bbox_color) in enumerate(zip(boxes, result_papago, colors)):
        # str_trns=translation_en2ko(text)
        # print(f'Papago responese : {str_trns}')
        if resp_code_papago==200:
            prediction.append(
                {
                    'translation':text,
                    'point':bbox.tolist(),
                    'color':bbox_color,}
                )
        else:
            prediction.append(
                {
                    'translation':f'Papago API Error [{resp_code_papago}]...',
                    'point':bbox.tolist()}
                )
    
    running_time = time.monotonic() - time_start
    print("*****", datetime.datetime.now(), "*****")
    print(f'inference time : {running_time:.2f}s')

    return prediction

def make_fots_response_no_split(image,time_start):
    boxes, pred_transcripts = predict(image)
    colors = color_list(image, boxes)
    print(f'Detect Text : {pred_transcripts}')
    prediction=[len(boxes)]
    for idx,(bbox,text,bbox_color) in enumerate(zip(boxes, pred_transcripts, colors)):
        resp_code_papago,result_papago = translation_en2ko(text)
        # str_trns=translation_en2ko(text)
        # print(f'Papago responese : {str_trns}')
        if resp_code_papago==200:
            prediction.append(
                {
                    'translation':result_papago,
                    'point':bbox.tolist(),
                    'color':bbox_color,}
                )
        else:
            prediction.append(
                {
                    'translation':f'Papago API Error [{resp_code_papago}]...',
                    'point':bbox.tolist()}
                )
    
    running_time = time.monotonic() - time_start
    print("*****", datetime.datetime.now(), "*****")
    print(f'inference time : {running_time:.2f}s')

    return prediction