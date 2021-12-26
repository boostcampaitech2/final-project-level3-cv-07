from PIL import Image
import json
import os
import os.path as osp
from glob import glob
import numpy as np
from torch.utils.data import DataLoader, Dataset
from cv2 import contourArea
# from shapely.geometry import Polygon
from tqdm import tqdm

IMAGE_EXTENSIONS = {'.gif', '.jpg', '.png'}
TEXT = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

class COCOConverter(Dataset):
    def __init__(self, images_dir, seg_dir):
        assert os.path.isdir(images_dir), f'dir \'{images_dir}\' not found!'
        self.image_paths = {x for x in glob(osp.join(images_dir, '*')) if osp.splitext(x)[1] in
                       IMAGE_EXTENSIONS}
        self.seg_paths = list(glob(osp.join(seg_dir, '*.txt')))

        self.data = {
            "images": [],
            "categories": [],
            "annotations": []
        }

    def add_image(self, filename, height, id, width):
        image = {
            "file_name": filename,
            "height": height,
            "id": id,
            "width": width            
        }
        self.data['images'].append(image)
    
    def get_categories(self, id, name, supercategory='text'):
        categories =  {
                    "id": id,
                    "name": name,
                    "supercategory": supercategory
                }
        self.data['categories'].append(categories)

    def add_annotation(self, area, bbox, category_id, id, image_id, iscrowd, segmentation):
        annotation = {
            "area": area,
            "bbox": bbox,
            "category_id": category_id,
            "id": id,
            "image_id": image_id,
            "iscrowd": iscrowd,
            "segmentation": segmentation
        }
        self.data['annotations'].append(annotation)

    def convert(self):
        print('start converting...')
        ann_id = 1
        image_id = 1
        category_id = 2
        self.get_categories(1, "text")

        for alphabet in TEXT:           
            self.get_categories(category_id, alphabet)
            category_id += 1
        for idx, img in enumerate(self.image_paths):
            f_name = os.path.basename(img)
            if os.path.isfile(img):
                im = Image.open(img)
                width, height = im.size
                image_name = str(f_name)
                # im.save(os.path.join(self.image_paths, f_name))
                # im.close()
                self.add_image(f_name, height, image_id, width)        
                seg_file = self.seg_paths[idx]
                # gt_file = self.gt_paths[idx]
                # voca_file = self.voca_paths[idx]
                with open(seg_file) as f:
                    id = 1
                    iscrowd = 0
                    for line in f.readlines():
                        if line:
                            line = line.strip()
                            check = line.split(" ")
                            if len(check) != 10 or check[-1][1] not in TEXT:
                                continue
                            char = check[-1][1]
                            cat_id = TEXT.index(char)+1
                            bbox = list(map(int, check[5:-1])) 
                            seg = [[bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]]
                            color = line[:3]
                            area = (bbox[2] - bbox[0])*(bbox[3]-bbox[1])
                            self.add_annotation(area, bbox, cat_id, id, image_id, iscrowd, seg)
                            id += 1
            image_id += 1
        with open("../train.json", "w") as f:
            json.dump(self.data, f, indent = 4)
                # with open(gt_file) as f:
                    
                #     for line in f.readlines():
                #         line = line.strip()
                #         line = line.split(" ")
                #         line = line.pop()
                #         box.append(line)

                # with open(voca_file) as f:
                #     voca = f.read().split("\n")
                #     study = []
                #     for v in voca:
                #         if v in gt:

                    # if not skip:
                    #     img_idx = label_file.split('_')[-1].split('.')[0]
                    #     image_path = os.path.join(icdar_images_dir, 'img_' + img_idx)
                    #     if os.path.isfile(image_path + '.jpg'):
                    #         im = Image.open(image_path + '.jpg')
                    #     elif os.path.isfile(image_path + '.png'):
                    #         im = Image.open(image_path + '.png').convert('RGB')
                    #     else:
                    #         raise RuntimeError(f'{img_idx}')
                    #     width, height = im.size
                    #     image_name = str(image_id) + '.jpg'
                    #     im.save(os.path.join(self.coco_images_dir, image_name))
                    #     im.close()
                    #     self.add_image(image_id, width, height, image_name)
                    #     image_id += 1
def main():
    images_dir = '../input_images/train/'
    seg_dir = '../input/seg/'
    x = COCOConverter(images_dir, seg_dir)
    x= x.convert()

if __name__ == "__main__":
    main()