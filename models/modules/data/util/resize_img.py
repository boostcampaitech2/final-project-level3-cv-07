import os
import sys
sys.path.append('/opt/ml/project/models')
from modules.data.utils import check_and_validate_polys
from modules.data.utils import crop_area
from modules.data.utils import generate_rbox
from itertools import compress

import numpy as np
import cv2
import pathlib


def load_gt(gt_names):
    all_bboxs = []
    all_texts = []
    for gt_name in gt_names:
        gt_src = os.path.join(gt_path, gt_name)
        with open(gt_src, 'r', encoding='utf-8') as f:
            bboxes = []
            texts = []
            for line in f:
                text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(',')
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, text[:8]))
                bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                delim, label = '', ''
                for i in range(9, len(text)):
                    label += delim + text[i]
                    delim = ','
                texts.append(label.strip())
                bboxes.append(bbox)
            bboxes = np.array(bboxes)
            all_bboxs.append(bboxes)
            all_texts.append(texts)
    return all_bboxs, all_texts

def transform(gt, input_size=512, crop=False, random_scale=np.array([0.5, 1, 2.0, 3.0]), background_ratio=3. / 8):
    """
    :param gt: iamge path (str), wordBBoxes (2 * 4 * num_words), transcripts (multiline)
    :return:
    """
    image_path, wordBBoxes, transcripts = gt
    im = cv2.imread(image_path.as_posix())
    numOfWords = len(wordBBoxes)
    text_polys = wordBBoxes  # num_words * 4 * 2
    # transcripts = [word for word in transcripts for word in line.split()]
    text_tags = [False if tag == '###' else True for tag in transcripts]  # ignore '###'    

    if numOfWords == len(transcripts):
        try:
            h, w, _ = im.shape
        except:
            print(image_path)
        text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

        rd_scale = 1 #np.random.choice(random_scale)
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        text_polys *= rd_scale

        rectangles = []

        if crop:
            im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background=True)
            transcripts = [transcripts[i] for i in selected_poly]

        # pad the image to the training input size or the longer side of image
        new_h, new_w, _ = im.shape
        max_h_w_i = np.max([new_h, new_w, input_size])
        im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
        im_padded[:new_h, :new_w, :] = im.copy()
        im = im_padded
        # resize the image to input size
        new_h, new_w, _ = im.shape
        resize_h = input_size
        resize_w = input_size
        im = cv2.resize(im, dsize=(resize_w, resize_h))
        resize_ratio_3_x = resize_w / float(new_w)
        resize_ratio_3_y = resize_h / float(new_h)
        text_polys[:, :, 0] *= resize_ratio_3_x
        text_polys[:, :, 1] *= resize_ratio_3_y
            
        new_h, new_w, _ = im.shape
        score_map, geo_map, training_mask, rectangles = generate_rbox((new_h, new_h), text_polys, text_tags)

        for rectangle_idx in range(len(rectangles)):      # erase wrong bbox annotation
            if rectangles[rectangle_idx][0] == '*':
                text_tags[rectangle_idx] = False
                print('Erase a wrong point in '+str(image_path)+' - line '+str(rectangle_idx))

        images = im[:, :, :].astype(np.float32)  # bgr -> rgb : im[:, :, ::-1].astype(np.float32)
        
        transcripts = list(compress(transcripts, text_tags))
        rectangles = list(compress(rectangles, text_tags))  # [ [pt1, pt2, pt3, pt3],  ]

        return image_path, images, transcripts, rectangles
    else:
        print(image_path)
        print(transcripts)
        raise TypeError('Number of bboxes is inconsist with number of transcripts ')




def make_resize_image(gt_path, img_path, size=512):
    gt_names = sorted(os.listdir(gt_path))[1:]
    img_names = sorted(os.listdir(img_path))[1:]
    
    if len(gt_names) != len(img_names):
        raise TypeError('Number of grount truths is inconsist with number of images ')

    new_gt_path = gt_path + '_resize'
    new_img_path = img_path + '_resize'
    if not os.path.isdir(new_gt_path):
        os.makedirs(new_gt_path)
    if not os.path.isdir(new_img_path):
        os.makedirs(new_img_path)

    all_bbox, all_texts = load_gt(gt_names)
    all_img_src = [pathlib.Path(os.path.join(img_path, img_name))  for img_name in img_names]
    for file_idx in range(len(img_names)):
        if img_names[file_idx].split('.')[-1] == 'gif':
            continue
        transform_result = transform((all_img_src[file_idx], all_bbox[file_idx], all_texts[file_idx]), input_size=size, crop=False)
        image_path, images, transcripts, rectangles = transform_result

        if len(transcripts) == 0:
            print('no text found in resized image : ', image_path)
        else:
            # image 생성
            image_path = os.path.join(new_img_path, img_names[file_idx])
            cv2.imwrite(image_path, images)

            # ground truth 생성
            gt_path = os.path.join(new_gt_path, gt_names[file_idx])
            gt = ''
            for line_idx in range(len(transcripts)):
                for location in rectangles[line_idx]:
                    gt += str(location)+','
                gt += 'OE_pos,'+transcripts[line_idx] + '\n'
            with open(gt_path,'w') as f:
                f.write(gt)
                
if __name__=='__main__':
    datasets_path = '/opt/ml/project/models/datasets'
    gt_path = datasets_path + '/train_gts'
    img_path = datasets_path + '/train_images'
    
    make_resize_image(gt_path, img_path, size=512)