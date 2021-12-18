import argparse
import json
import os, sys, copy
import subprocess

import numpy as np
import PIL.ImageDraw
import PIL.ImageFont
import torch

import assessors
import generators
import transformations.pytorch as transformations
import utils.common
import utils.pytorch
from evaluator import MeanAssessorScoreEvaluator

import matplotlib.pylab as plt
import seaborn as sns

from scipy.spatial import distance

np.random.seed(seed=999)


def get_direction_and_score(z, y, step_size, transform, which_way):
    if transform:
        z_transformed = transformation.transform_test(z, y, step_size, which_way)
        z_transformed = z.norm() * z_transformed / z_transformed.norm()
        z = z_transformed
    gan_images = utils.pytorch.denorm(generator(z, y))
    gan_images = input_transform(gan_images)
    gan_images = gan_images.view(-1, *gan_images.shape[-3:])
    gan_images = gan_images.to(device)
    out_scores_current = output_transform(assessor(gan_images))
    out_scores_current = out_scores_current.detach().cpu().numpy()
    if len(out_scores_current.shape) == 1:
        out_scores_current = np.expand_dims(out_scores_current, 1)
    return z, out_scores_current


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='path for directory with the checkpoints of the trained model we want to use')
    parser.add_argument('--checkpoint', type=int, required=True, help='which checkpoint to load')
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()
    opts = vars(args)

    gpu_id = 0

    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

    train_opts_file = os.path.join(opts["checkpoint_dir"], "opts.json")
    with open(train_opts_file) as f:
        train_opts = json.load(f)

    if not isinstance(train_opts["transformer"], list):
        train_opts["transformer"] = [train_opts["transformer"]]

    checkpoint_dir = opts["checkpoint_dir"]
    result_dir = opts["out"]
    os.makedirs(result_dir, exist_ok=True)

    # Saving testing settings
    opts_file = os.path.join(result_dir, "opts.json")
    with open(opts_file, 'w') as fp:
        json.dump(opts, fp)

    dim_z = {
        'biggan256': 140,
        'biggan512': 128
    }.get(train_opts['generator'][0])

    vocab_size = {'biggan256': 1000, 'biggan512': 1000}.get(train_opts['generator'][0])

    transformer = train_opts["transformer"][0]
    transformer_arguments = train_opts["transformer"][1]
    if transformer_arguments != "None":
        key_value_pairs = transformer_arguments.split(",")
        key_value_pairs = [pair.split("=") for pair in key_value_pairs]
        transformer_arguments = {pair[0]: pair[1] for pair in key_value_pairs}
    else:
        transformer_arguments = {}

    transformation = getattr(transformations, transformer)(dim_z, vocab_size, **transformer_arguments)
    transformation = transformation.to(device)

    generator = train_opts["generator"][0]
    generator_arguments = train_opts["generator"][1]
    if generator_arguments != "None":
        key_value_pairs = generator_arguments.split(",")
        key_value_pairs = [pair.split("=") for pair in key_value_pairs]
        generator_arguments = {pair[0]: pair[1] for pair in key_value_pairs}
    else:
        generator_arguments = {}

    generator = getattr(generators, generator)(**generator_arguments)

    for p in generator.parameters():
        p.requires_grad = False
    generator.eval()
    generator = generator.to(device)

    assessor_elements = getattr(assessors, train_opts['assessor'])(True)
    if isinstance(assessor_elements, tuple):
        assessor = assessor_elements[0]
        input_transform = assessor_elements[1]
        output_transform = assessor_elements[2]
    else:
        assessor = assessor_elements
        def input_transform(x): return x  # identity, no preprocessing
        def output_transform(x): return x  # identity, no postprocessing

    if hasattr(assessor, 'parameters'):
        for p in assessor.parameters():
            p.requires_grad = False
            assessor.eval()
            assessor.to(device)

    transformation.load_state_dict(torch.load(os.path.join(checkpoint_dir, "pytorch_model_" + str(opts["checkpoint"]) + ".pth")))

    # Test settings
    truncation = 1
    num_samples = 1
    iters = 5
    alpha = 0.1

    num_categories = 100

    ways = [{"score":[], "direction":None, "distances":[]} for i in range(transformation.direction_num)]

    for y in range(num_categories):

        zs = utils.common.truncated_z_sample(num_samples, dim_z, truncation)
        ys = np.repeat(y, num_samples)
        zs = torch.from_numpy(zs).type(torch.FloatTensor).to(device)
        ys = torch.from_numpy(ys).to(device)
        ys = utils.pytorch.one_hot(ys, vocab_size)

        step_sizes = np.repeat(np.array(alpha), num_samples * dim_z).reshape([num_samples, dim_z])
        step_sizes = torch.from_numpy(step_sizes).type(torch.FloatTensor).to(device)
        feed_dicts = []
        for batch_start in range(0, num_samples, 4):
            s = slice(batch_start, min(num_samples, batch_start + 4))
            feed_dicts.append({"z": zs[s], "y": ys[s], "truncation": truncation, "step_sizes": step_sizes[s]})

        for feed_dict in feed_dicts:
            for way in range(transformation.direction_num):
                z_start = feed_dict["z"]
                step_sizes = feed_dict["step_sizes"]
                z_next = z_start
                initial_score = None
                for iter in range(0, iters, 1):
                    feed_dict["step_sizes"] = step_sizes
                    feed_dict["z"] = z_next
                    tmp, outscore = get_direction_and_score(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], transform=True, which_way=way)
                    if initial_score is None:
                        initial_score = outscore
                    z_next = tmp
                ways[way]["score"].append(outscore.item()-initial_score.item())
                ways[way]["direction"] = (z_next-z_start).detach().cpu().numpy().reshape(-1)
            
            for i in range(len(ways)):
                ways[i]["distances"].append([])
                for j in range(len(ways)):
                    ways[i]["distances"][-1].append(distance.cosine(ways[i]["direction"], ways[j]["direction"]))

    for way in range(len(ways)):
        ways[way]["score"] = np.mean(ways[way]["score"])
        ways[way]["distances"] = np.mean(ways[way]["distances"], axis=0)
    

    with open( os.path.join(result_dir, "distances.txt"), 'w+') as f:
        for way in range(len(ways)):
            f.write(f"Total step size={alpha*iters} (iterative with alpha={alpha})\n")
            f.write(f"Avg score increase for {way}: {ways[way]['score']}\n")
            for other_way in range(len(ways)):
                f.write(f"cosine dist({way} - {other_way}): {ways[way]['distances'][other_way]}\n")
            f.write('\n')