from modules.utils.util import resize_image
import torch
import numpy as np
import json

from modules.utils.converter import keys
from modules.utils.converter import StringLabelConverter
from modules.models.model import OCRModel

converter = StringLabelConverter(keys)

model = None

def load_model(model_path, with_gpu=None):
    config = json.load(open('config.json'))
    checkpoints = torch.load(model_path, map_location='cpu')
    if not checkpoints:
        raise RuntimeError('No checkpoint found.')
    print('Epochs: {}'.format(checkpoints['epoch']))
    state_dict = checkpoints['state_dict']
    model = OCRModel(config)
    model.load_state_dict(state_dict)
    if with_gpu:
        model.to(torch.device('cuda'))
    model.eval()
    return model

def load_model_at_run():
    print("start model load...")
    global model
    if model is None:
        with_gpu = True if torch.cuda.is_available() else False
        model_path = 'saved/new/model_best.pth.tar'
        model = load_model(model_path, with_gpu)

@torch.no_grad()
def predict(im):
    h, _, _ = im.shape
    global model
    with_gpu = True if torch.cuda.is_available() else False
    if model is None:
        model_path = 'saved/new/model_best.pth.tar'
        model = load_model(model_path, with_gpu)

    im_resized, (ratio_h, ratio_w) = resize_image(im)
    im_resized = im_resized.astype(np.float32)
    im_resized = torch.from_numpy(im_resized)
    if with_gpu:
        im_resized = im_resized.cuda()

    im_resized = im_resized.unsqueeze(0)
    im_resized = im_resized.permute(0, 3, 1, 2)

    score, geometry, preds, boxes, mapping, rois = model.forward(im_resized, None, None)

    img = np.array(im).astype(np.uint8)
    img = img[:, :, ::-1]
    pred_transcripts=[]
    if len(boxes) != 0:
        scores = boxes[:, 8].reshape(-1)
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h
        # boxes[:, :, 1] = h - boxes[:, :, 1]
        # decode predicted text
        pred, preds_size = preds
        _, pred = pred.max(2)
        pred = pred.transpose(1, 0).contiguous().view(-1)
        pred_transcripts = converter.decode(pred.data, preds_size.data, raw=False)
        pred_transcripts = [pred_transcripts] if isinstance(pred_transcripts, str) else pred_transcripts
        
    return boxes, pred_transcripts

