import os, sys, glob, argparse
import cv2
import numpy as np

sys.path.append("../pytorch")

from evaluator import MeanAssessorScoreEvaluator


parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', type=str, required=True)
parser.add_argument('--which_way', type=int, default=0, help='if one_way is 1, which way you want to calculate')
parser.add_argument('--is_tf', type=int, default=0, help='if it is tensorflow -> 1, otherwise 0')
args = parser.parse_args()

## PARAMETERS ##
folder_path = args.folder_path
which_way = args.which_way 
################

images = glob.glob(os.path.join(folder_path, "*.jpg"))

def partition_image(img):
    rows = []
    start_idx = 0
    while (start_idx+256<=img.shape[0]):
        row = img[start_idx:start_idx+256,:,:]
        start_idx+=257
        rows.append(row)
    return rows


def extract_images(row):
    images = []
    start_idx = 0
    while (start_idx+256<=row.shape[1]):
        img = row[:, start_idx:start_idx+256,:]
        start_idx+=257
        images.append(img)
    return images


alphas = list(np.linspace(0.1*-5, 0.1*5, 11))
evaluator = MeanAssessorScoreEvaluator(alpha_values=list(np.linspace(0.1*-5, 0.1*5, 11)))

for img_path in images:
    if (args.is_tf == 1) and (int(img_path[-5])!=args.which_way):
        continue
    whole_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    rows = partition_image(whole_img)

    for row in rows:
        imgs = extract_images(row)
        for alpha_idx, img in enumerate(imgs):
            alpha = alphas[alpha_idx]
            evaluator.update(alpha, 0, np.array([img]))

evaluator.finish("assessor_score_plot.png", folder_path, which_way, True)
