import os
import shutil
import glob
import re
import cv2
from tqdm import tqdm

if __name__ == '__main__':
    datasets_path = 'datasets/test_2017'
    file_path = datasets_path + '/train_gts_ori'
    file_names = sorted(os.listdir(file_path))
    image_path = datasets_path + '/train_images_ori'
    
    new_file_path = file_path + '_alphabet'
    new_image_path = image_path + '_alphabet'
    if not os.path.isdir(new_file_path):
        os.makedirs(new_file_path)
    if not os.path.isdir(new_image_path):
        os.makedirs(new_image_path)
        
    regular_char = re.compile(r"[ 0-9a-zA-Z@!#$%&'()+,-.·/:;=?´_~*°<>\"\'\[\]]")
    for file_name in tqdm(file_names):
        src = os.path.join(file_path, file_name)
        try:
            with open(src, 'r', encoding='utf-8') as f:
                new_texts = ''
                lines = f.readlines()
                eng_exist = False
                for line in lines:
                    texts = line[:-1].split(',')
                    bbox = ','.join(texts[:8])
                    script = texts[8]
                    trans = ','.join(texts[9:])

                    possible_word = False
                    if script == 'Latin':
                        possible_word = True
                        for char in trans:
                            if not regular_char.match(char):
                                possible_word = False

                    if trans == '':
                        continue
                    elif possible_word:
                        eng_exist = True
                        new_texts += f"{bbox},{script},{trans}\n"
                    else:
                        new_texts += bbox+',None,###\n'
        except:
            print(file_name)
            continue
        image_name = 'img_' + file_name.split('_')[2].split('.')[0]+'.*'
        image_name = glob.glob(image_path+'/'+image_name)[0].split('/')[-1]
        image_src = os.path.join(image_path, image_name)
        img = cv2.imread(image_src)
        # 가로 세로 800 미만인 이미지만 test에 사용
        if eng_exist and img.shape[0] < 800 and img.shape[1] < 800:
            new_src = os.path.join(new_file_path, file_name)
            with open(new_src,'w') as f:
                f.write(new_texts)
            new_image_src = os.path.join(new_image_path, image_name)
            shutil.copyfile(image_src, new_image_src)
        
    