import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import copy
from skimage import color, measure
import matplotlib.pyplot as plt

class MaskRCNN(object):
    def __init__(self,  process_size=256):
        self.process_size = process_size
        transforms = []
        transforms.append(T.ToPILImage())
        transforms.append(T.Resize(process_size))
        transforms.append(T.ToTensor())
        self.transform = T.Compose(transforms)
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to("cuda")
        self.model.eval()

    def inference(self, input):
        img = copy.deepcopy(input)
        if isinstance(img, str):
            img = np.array(Image.open(img).convert("RGB").resize((self.process_size,self.process_size)))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img).to("cuda")
        predictions = self.model(img.unsqueeze(0))[0]
        return self.postprocess(predictions)

    def postprocess(self, preds, imwrite_path=None):
        if len(preds["scores"]) == 0:
            return None
        mask_idx = np.argmax(preds["scores"].detach().cpu().numpy())
        mask = preds["masks"][mask_idx].squeeze().detach().cpu().numpy()
        #print("mask", mask.shape, "min", mask.min(), "max", mask.max(), "mean", mask.mean())
        if imwrite_path:
            mask_write = Image.fromarray(mask*255).convert("L")
            mask_write.save(imwrite_path)
        mask = mask>0.5
        if mask.sum()==0:
            return None

        label_image = measure.label(mask)
        props = measure.regionprops(label_image)[0]
        return props
    

def calculate_features(input, rcnn):
    """
    Parameters
    img (np.array): 0-255, RGB, float
    
    Return
    list[float]: colorfulness, brightness, redness, entropy
    """ 
    img = copy.deepcopy(input)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # gan output is RGB, we need BGR
    img = cv2.resize(img, (256,256)) # to speed up the computation
    props = rcnn.inference(img)
    if props is None:
        return colorfulness(img), brightness(img), redness(img), entropy(img), None, None, None

    try:
        return colorfulness(img), brightness(img), redness(img), entropy(img), centerness(props), object_size(props), squareness(props)
    except ZeroDivisionError:
        print("ZeroDivision during MaskRCNN inference")
        return colorfulness(img), brightness(img), redness(img), entropy(img), None, None, None

def object_size(props):
    return props.area


def squareness(props):
    return props.minor_axis_length / props.major_axis_length


def centerness(props, image_size=256):
    res = sum([abs(image_size // 2 - c) for c in props.centroid])
    return 1 - (res / image_size)


# taken from pyimagesearch
def colorfulness(input):
    """
    Parameters
    img (cv:MatArray): 0-255, BGR, uint8
    
    Return
    float: colorfulness
    """ 
    image = copy.deepcopy(input)
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


def brightness(input):
    """
    Parameters
    img (cv:MatArray): 0-255, BGR, uint8
    
    Return
    float: brightness value between 0-1
    """ 
    image = copy.deepcopy(input)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray_img)/255.


def redness(input):
    image = copy.deepcopy(input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_img = color.rgb2hsv(image)
    h, s, v = hsv_img[..., 0], hsv_img[..., 1], hsv_img[..., 2]
    ones = np.ones_like(h)
    zeros = np.zeros_like(h)
    red = np.where(np.logical_or(h < 0.1, h > 0.9), ones, zeros)
    red = np.where(np.logical_and(red, s > 0.5), ones, zeros)
    red = np.where(np.logical_and(red, v > 0.2), ones, zeros)
    out = np.sum(red) / np.prod([image.shape[:2]])
    return out


def entropy(input):
    """
    Parameters
    img (cv:MatArray): 0-255, BGR, uint8
    
    Return
    float: entropy
    """
    img = copy.deepcopy(input)
    img = img.astype("float")
    img = np.reshape(img, -1)
    hist1 = np.histogram(img, bins=int(img.max()+1), density=True)
    data = hist1[0] + 1e-10 
    ent = -(data*np.log(np.abs(data))).sum()
    return ent



if __name__ == "__main__":
    rcnn = MaskRCNN()
    img1 = cv2.cvtColor(cv2.imread("1.jpg"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("2.jpg"), cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(cv2.imread("3.jpg"), cv2.COLOR_BGR2RGB)
    img4 = cv2.cvtColor(cv2.imread("plane.jpg"), cv2.COLOR_BGR2RGB)
    print(calculate_features(img1, rcnn))
    print(calculate_features(img2, rcnn))
    print(calculate_features(img3, rcnn))
    print(calculate_features(img4, rcnn))
    