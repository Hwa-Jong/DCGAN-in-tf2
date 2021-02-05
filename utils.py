###Version 1.3

import os
import glob
import cv2
import numpy as np

ERASE_LINE = '\x1b[2K' # erase line command

SPLIT_LINE = '\n#------------------------------------------------------------------------------------------------------------------------------------'

def BGRtoRGB(imgs):
    imgs = imgs[...,::-1]
    return imgs

def load_images_in_folder(fpath, bgr2rgb=False, img_size=None):    
    img_names = glob.glob(os.path.join(fpath, '*.jpg'))
    imgs = []
    for name in img_names:
        imgs.append(cv2.imread(name))
        if img_size != None:
            img = cv2.resize(img, dsize=img_size)
    imgs = np.array(imgs)

    if bgr2rgb:
        imgs = BGRtoRGB(imgs)
    
    return imgs


class my_log():
    def __init__(self, path):
        self.__path = path
        
    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, path):
        self.__path = path

    def logging(self, context, log_only=False):
        if not os.path.exists(self.__path):
            f = open(self.__path, 'w')
        else:
            f = open(self.__path, 'a')

        if isinstance(context, list):
            for data in context:
                f.write(str(data)+'\n')
                if not log_only:
                    print(data)
        elif isinstance(context, str):
            f.write(context+'\n')
            if not log_only:
                print(context)
        
        f.close()

    def show(self, apply_split=False):
        if not os.path.exists(self.__path):
            print('not exist file!! - path : %s' %self.__path)
            return None
        else:
            f = open(self.__path, 'r')
            text = []
            while True:
                line = f.readline()
                if not line: break
                line = line.strip()
                if apply_split:
                    line = line.split()
                print(line)
                text.append(line)
            f.close()

            return text
