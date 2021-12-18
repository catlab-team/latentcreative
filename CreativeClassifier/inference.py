import os, argparse, shutil, sys, time, glob
from datetime import datetime

import torch
from torch.utils.data.dataloader import DataLoader
from efficientnet_pytorch import EfficientNet

from core.dataset import *


torch.backends.cudnn.benchmark = True


def main(args, images):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = CreativeTestDataset(images, args.input_size)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                            shuffle=False, pin_memory=True, \
                            num_workers=0)

    model = EfficientNet.from_pretrained(args.arch, num_classes=1)
    model = torch.load(args.checkpoint, map_location="cpu")
    model = model.to(device)
    model.eval()
        
    with torch.no_grad():
        for im_name, images in test_loader:
            images = images.to(device)
            output = torch.nn.functional.softmax(model(images))
            print(im_name[0], output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", required=True, help="Input Folder Path")
    parser.add_argument("--checkpoint", required=True, help="ckpt path")

    parser.add_argument("--arch", default="efficientnet-b0", help="efficientnet arch")
    parser.add_argument("--input-size", default=224, help="Input Image Size")
    parser.add_argument("--batch-size", default=1, help="Batch Size")

    args = parser.parse_args()
    exts = ["jpg", "png", "jpeg", "bmp", "JPG", "PNG", "JPEG", "BMP",]
    input_images = []
    for ext in exts:
        input_images.extend(glob.glob(os.path.join(args.input_folder, "*."+ext)))
    main(args, input_images)
