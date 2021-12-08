import time
from fastapi import UploadFile, File
from typing import List
import base64

import datetime
from server.modules.papago import translation_en2ko
from server.modules.util import read_imagefile,read_imagebase
from server.modules.inference import predict
from api import app

@app.post("/fots/image")
async def fots_image(file: UploadFile = File(...)):
    time_start = time.monotonic()
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        print(datetime.datetime.now())
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    boxes, pred_transcripts = predict(image)
    print(f'Detect Text : {pred_transcripts}')
    join_str=" @ "
    print(join_str.join(pred_transcripts))
    resp_code_papago,result_papago = translation_en2ko(join_str.join(pred_transcripts))
    print(result_papago)
    result_papago=result_papago.split(join_str)
    print(f'Translated Text : {result_papago}')
    prediction=[len(boxes)]
    for idx,(bbox,text) in enumerate(zip(boxes, result_papago)):
        # str_trns=translation_en2ko(text)
        # print(f'Papago responese : {str_trns}')
        if resp_code_papago==200:
            prediction.append(
                {
                    'translation':text,
                    'point':bbox.tolist()}
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



@app.post("/fots/base64")
async def fots_base64(file: UploadFile = File(...)):
    time_start = time.monotonic()
    image = read_imagefile(base64.b64decode(await file.read()))
    boxes, pred_transcripts = predict(image)
    print(f'Detect Text : {pred_transcripts}')
    join_str=" @ "
    print(join_str.join(pred_transcripts))
    resp_code_papago,result_papago = translation_en2ko(join_str.join(pred_transcripts))
    print(result_papago)
    result_papago=result_papago.split(join_str)
    print(f'Translated Text : {result_papago}')
    prediction=[len(boxes)]
    for idx,(bbox,text) in enumerate(zip(boxes, result_papago)):
        # str_trns=translation_en2ko(text)
        # print(f'Papago responese : {str_trns}')
        if resp_code_papago==200:
            prediction.append(
                {
                    'translation':text,
                    'point':bbox.tolist()}
                )
        else:
            prediction.append(
                {
                    'translation':f'Papago API Error [{resp_code_papago}]...',
                    'point':bbox.tolist()}
                )
    running_time = time.monotonic() - time_start
    print(datetime.datetime.now())
    print(f'inference time : {running_time:.2f}s')

    return prediction

@app.post("/fots/image/nopapago")
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

@app.post("/fots/base64/nopapago")
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