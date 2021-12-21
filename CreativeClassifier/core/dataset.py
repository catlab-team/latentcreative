import os, glob
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from .utils import AddGaussianNoise


def read_anc_data(path):
    with open(path, "r") as file:
        lines = file.readlines()
    data = []
    for line in lines:
        key, num_anc = line.split(" ")[0], int(line.split(" ")[1])
        data.append(key)
    return data

class CreativeDataset(Dataset):
    def __init__(self, path, debug, data_cfg, aug=True, is_train=True):
        self.data_cfg = data_cfg
        self.creative_images = read_anc_data(os.path.join(path, data_cfg.creative_data))
        self.zero_anc_images = read_anc_data(os.path.join(path, data_cfg.random_data))
        if is_train:
            self.biggan_generated = glob.glob(os.path.join(self.data_cfg.biggan_generated_train, "*.jpg"))
        else:
            self.biggan_generated = glob.glob(os.path.join(self.data_cfg.biggan_generated_val, "*.jpg"))
        if debug:
            end_idx = 1500 if is_train else 150
            self.images = [(os.path.join(self.data_cfg.artbreeder_folder, "{}.jpeg".format(im_path)), 1) for im_path in self.creative_images][:end_idx*2]
            self.images += [(os.path.join(self.data_cfg.artbreeder_folder, "{}.jpeg".format(im_path)), 0) for im_path in self.zero_anc_images][:end_idx]
            self.images += [(im_path, 0) for im_path in self.biggan_generated][:end_idx]
        else:
            creative_end_idx = int(48000*0.8) if is_train else int(48000*0.2)
            self.images = [(os.path.join(self.data_cfg.artbreeder_folder, "{}.jpeg".format(im_path)), 1) for im_path in self.creative_images[:creative_end_idx]]
            curr_len = len(self.images)
            print(is_train, "creative",curr_len)
            self.images += [(os.path.join(self.data_cfg.artbreeder_folder, "{}.jpeg".format(im_path)), 0) for im_path in self.zero_anc_images]
            print(is_train, "zero_anc",len(self.images)-curr_len)
            curr_len = len(self.images)
            self.images += [(im_path, 0) for im_path in self.biggan_generated]
            print(is_train, "biggan",len(self.images)-curr_len)

        if aug:
            self.transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(data_cfg.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    AddGaussianNoise(0., 4./255.)])
        else:
            self.transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(data_cfg.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        if is_train:
            np.random.shuffle(self.images)
            print("Train Dataset: {}".format(len(self.images)))
        else:
            print("Val Dataset: {}".format(len(self.images)))

    def __getitem__(self, index):
        im_path, label = self.images[index]
        img = cv2.imread(im_path.strip())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transforms(img), label

    def __len__(self):
        return len(self.images)

class CreativeTestDataset(Dataset):
    def __init__(self, input_images, im_size):
        self.images = input_images
        self.transforms = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(im_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        im_path = self.images[index]
        img = cv2.imread(im_path.strip())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return im_path, self.transforms(img)

    def __len__(self):
        return len(self.images)
