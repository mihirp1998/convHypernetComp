# modified from https://github.com/desimone/vision/blob/fb74c76d09bcc2594159613d5bdadd7d4697bb11/torchvision/datasets/folder.py
import cv2
import os
import numpy as np
import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import pickle
import random
from collections import defaultdict
import glob
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def readImage(path):
    img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255.0
    return img

def default_loader(paths,root):
    paths = [os.path.join(root,i) for i in paths]
    imgs  = [readImage(path) for path in paths]
    imgs = np.stack(imgs, axis=0)
    return imgs

def crop_cv2(img, patchx,patchy):
    b,c,height, width = img.shape
    #print(height)
    start_x = random.randint(0, height - patchx)
    start_y = random.randint(0, width - patchy)
    # start_x = 50
    # start_y = 50
    return img[:,:,start_x : start_x + patchx, start_y : start_y + patchy]

def np_to_torch(img):
    img = np.swapaxes(img, 1, 2) #w, h, 9
    img = np.swapaxes(img, 1, 3) #9, h, w
    return torch.from_numpy(img).float()

def swap(img):
    img = np.swapaxes(img, 0, 2) #w, h, 9
    img = np.swapaxes(img, 0, 1) #9, h, w
    return img


class ImageFolder(data.Dataset):
    """ ImageFolder can be used to load images where there are no labels."""
    
    def genIds(self):
        vid_freq = dict()
        for i in self.imgs:
            imgtocomp = i[0]
            id_img = imgtocomp.split("/")[1][:30]
            # print(id_img)
            try:
                vid_freq[id_img] += 1
            except:
                vid_freq[id_img] = 1
        vid = 0
        vid2id = dict()        
        for w, c in vid_freq.items():
            vid2id[w] = vid
            vid +=1
        return vid_freq,vid2id

    def __init__(self, root,file_name = "trainHyperTuple100.p", train=True, loader=default_loader):
        images = []
        pickledFile = os.path.join(root,file_name)
        images = pickle.load(open(pickledFile,"rb"))

        self.d = defaultdict(lambda: [])
        self.id_names = []
        for i in images:
            id_val = i[0].split("/")[1][:-9]
            self.d[id_val].append(i[0])
            self.id_names.append(id_val)
        self.id_names = list(set(self.id_names))
        #print(images)
        self.root = root
        self.imgs = images
        self.loader = loader
        self.train= train
        #print(self.d)
        #self.vid_freq,self.vid2id = self.genIds()
        #pickle.dump(self.vid2id,open("train_dict1.p","wb"))

        self.vid2id = pickle.load(open("train_dict.p","rb"))
        self.vid_count = len(self.vid2id)
        print("vid count ",self.vid_count,len(self.id_names))

        #print("ids ",self.vid2id['framezg_14kFY2OM_000009_000019'])#should be 94

    def __getitem__(self, index):
        id_name = self.id_names[index]
        id_num = self.vid2id[id_name]
        #print(self.d[id_name])
        random.shuffle(self.d[id_name])
        files = self.d[id_name][:2]
        #print(id_name,files)
        imgs = self.loader(files,self.root)

        imgs = np_to_torch(imgs)

        # image = imgs[:3]
        if self.train:
            imgs = crop_cv2(imgs,128,128)
        # image = imgs[:3]
        return imgs,id_num,id_name

    def __len__(self):
        return len(self.id_names)


class face_forensics_dataloader(data.Dataset):
    def __init__(self, data_path, mode, transform=None):
        # /projects/katefgroup/datasets/faceforensics/original_sequences/youtube/c23/images
        self.data_path = data_path
        self.rgb_paths = []
        filepath = os.path.join(self.data_path, mode+".txt")
        fi = open(filepath, "r")
        for line in fi.readlines():
            line = line.strip()
            img_dir_path = os.path.join(self.data_path, line)
            rgb_paths = [img for img in glob.glob(img_dir_path + '/*.png')]
            self.rgb_paths.extend(rgb_paths)

        # self.rgb_paths = self.rgb_paths[0:1]
        print("Found %s rgb images" %len(self.rgb_paths))
        if transform != None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

        
    def __len__(self):
        return len(self.rgb_paths)
    
    def __getitem__(self, index):

        img = Image.open(self.rgb_paths[index]).convert('RGB')
        img = self.transform(img)
        img = img.float()
        return img
        






