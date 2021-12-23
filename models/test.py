import os
import json
import torch
import logging
import pathlib
import traceback
import argparse

import torch.utils.data as torchdata

from modules.utils.util import make_dir
from modules.utils.util import predict

from modules.utils.converter import keys
from modules.utils.converter import StringLabelConverter

from modules.models.model import OCRModel
from modules.models.metric import icdar_metric

from modules.data.utils import collate_fn
from modules.data.dataset_test import ICDAR

import numpy as np

logging.basicConfig(level=logging.DEBUG, format='')


def load_model(model_path, with_gpu):
    config = json.load(open('config.json'))
    logger.info("Loading checkpoint: {} ...".format(model_path))
    checkpoints = torch.load(model_path, map_location='cpu')
    if not checkpoints:
        raise RuntimeError('No checkpoint found.')
    print('Epochs: {}'.format(checkpoints['epoch']))
    state_dict = checkpoints['state_dict']
    model = OCRModel(config)
    if with_gpu and torch.cuda.device_count() > 1:
        model.parallelize()
    model.load_state_dict(state_dict)
    if with_gpu:
        model.to(torch.device('cuda'))
    model.eval()
    return model

def _to_tensor(*tensors):
    t = []
    device = ''
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
                
    for __tensors in tensors:
        
        t.append(__tensors.to(device))
    return t

def _eval_metrics(pred, gt):
    precious, recall, hmean = icdar_metric(pred, gt)
    return np.array([precious, recall, hmean])

def tester(model, dataloader):
    model.eval()
    total_val_metrics = np.zeros(3)
    test_data_loader = dataloader
    label_converter = StringLabelConverter(keys)
    
    with torch.no_grad():
        for batch_idx, gt in enumerate(test_data_loader):
            try:
                imagePaths, img, score_map, geo_map, training_mask, transcripts, boxes, mapping = gt
                img, score_map, geo_map, training_mask = _to_tensor(img, score_map, geo_map, training_mask)

                pred_score_map, pred_geo_map, pred_recog, pred_boxes, pred_mapping, rois = model.forward(
                    img, boxes, mapping)
                pred_transcripts = []
                pred_fns = []
                if len(pred_mapping) > 0:
                    pred_fns = [imagePaths[i] for i in pred_mapping]
                    pred, preds_size = pred_recog
                    _, pred = pred.max(2)
                    pred = pred.transpose(1, 0).contiguous().view(-1)
                    pred_transcripts = label_converter.decode(pred.data, preds_size.data, raw=False)
                    pred_transcripts = [pred_transcripts] if isinstance(pred_transcripts, str) else pred_transcripts
                pred_transcripts = np.array(pred_transcripts)

                gt_fns = [imagePaths[i] for i in mapping]
                total_val_metrics += _eval_metrics((pred_boxes, pred_transcripts, pred_fns),
                                                        (boxes, transcripts, gt_fns))

            except Exception:
                print(imagePaths)
                raise

    return {
        'test/precious': total_val_metrics[0] / len(test_data_loader),
        'test/recall': total_val_metrics[1] / len(test_data_loader),
        'test/hmean': total_val_metrics[2] / len(test_data_loader)
    }

def make_image(output_dir, input_image, model, with_gpu):
    make_dir(os.path.join(output_dir, 'img'))
    types = ('*.jpg', '*.png', '*.JPG', '*.PNG')  # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(input_image.glob(files))
    for image_fn in files_grabbed:
        try:
            with torch.no_grad():
                print('start predict')
                ploy, im = predict(image_fn, model, True, output_dir, with_gpu)
                print(image_fn, len(ploy))
        except Exception as e:
            print('excepted')
            traceback.print_exc()
            print(image_fn)
    
def main(args: argparse.Namespace):
    model_path = args.model
    output_dir = args.output_dir
    data_root = args.data_root
    input_size = args.input_size
    save_data = args.save_data
    
    with_gpu = True if torch.cuda.is_available() else False
    input_image = data_root / 'test_images'

    model = load_model(model_path, with_gpu)
    if save_data :
        make_image(output_dir, input_image, model, with_gpu)
            
    icdar_dataset = ICDAR(data_root, input_size)
    print(f'Length of dataset : {len(icdar_dataset)}')
    test_dataloader = torchdata.DataLoader(icdar_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                                         shuffle=False, collate_fn=collate_fn)
    print('make dataloader success')
    precious, recall, hmean = tester(model, test_dataloader).values()
    print(f'result : pricious-{precious}, recall-{recall}, hmean-{hmean}')

if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Model test')
    parser.add_argument('-m', '--model', default='saved/CRNN/model_best.pth.tar', type=pathlib.Path, required=False, help='path to model')
    parser.add_argument('-o', '--output_dir', default='output/test', type=pathlib.Path, help='output dir for drawn images')
    parser.add_argument('-d', '--data_root', default='datasets', type=pathlib.Path, required=False, help='dir for input image')
    parser.add_argument('-b', '--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('-n', '--num_workers', default=2, type=int, help='num worker')
    parser.add_argument('-i', '--input_size', default=512, type=int, help='input image size')
    parser.add_argument('-s', '--save_data', default=True, type=bool, help='determine save data')
    args = parser.parse_args()
    main(args)
