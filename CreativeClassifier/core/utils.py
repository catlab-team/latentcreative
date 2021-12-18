import numpy as np
import torch
import time
from tqdm import tqdm
import cv2
import math
import sys
import yaml
from easydict import EasyDict as edict


def time_to_str(time):
	hours, minutes, seconds = 0, 0, 0
	if time > 3600:
		hours = int(time / 3600)
		r1 = int(time) % 3600
		if r1 > 60:
			minutes = int(r1 / 60)
			r2 = int(r1) % 60
			seconds = r2
		else:
			seconds = r1
	else:
		if time > 60:
			minutes = int(time / 60)
			seconds = int(time) % 60
		else:
			seconds = int(time)
	time_str = ""
	if hours != 0:
		time_str += str(hours) + "h "
	if minutes != 0:
		time_str += str(minutes) + "m "
	if seconds != 0:
		time_str += str(seconds) + "s "
	return time_str[:-1]




def read_config(path):
	with open(path, 'r') as stream:
		config = yaml.safe_load(stream)
	cfg = edict(config)
	return cfg

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
