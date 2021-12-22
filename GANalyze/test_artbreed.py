import argparse
import json
import os
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

import glob

# Collect command line arguments
# --------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='which gpu to use.')
parser.add_argument('--alpha', type=float, default=0.1, help='stepsize for testing')
parser.add_argument('--test_truncation', type=float, default=1, help='truncation to use in test phase')
parser.add_argument('--checkpoint_dir', type=str, default="", help='path for directory with the checkpoints of the trained model we want to use')
parser.add_argument('--checkpoint', type=int, default=400000, help='which checkpoint to load')
parser.add_argument('--mode', default="bigger_step", choices=["iterative", "bigger_step"],
                    help="how to make the test sequences. bigger_step was used in the paper.")

parser.add_argument('--one_way', type=int, default=0, help='0 -> default 1-> if multiway, just one way result')
parser.add_argument('--which_way', type=int, default=0, help='if one_way is 1, which way you want to calculate, -1-> interpolation experiment')

parser.add_argument('--class_direction', type=int, default=1, help='whether class will change or not')
parser.add_argument('--seed', type=int, default=0, help='whether class will change or not')

args = parser.parse_args()
opts = vars(args)
print(opts)

# Choose GPU
if opts["gpu_id"] != -1:
    device = torch.device("cuda:" + str(opts["gpu_id"]) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# Creating directory to store output visualizations
train_opts_file = os.path.join(opts["checkpoint_dir"], "opts.json")
with open(train_opts_file) as f:
    train_opts = json.load(f)

if not isinstance(train_opts["transformer"], list):
    train_opts["transformer"] = [train_opts["transformer"]]

test_version = str(subprocess.check_output(["git", "describe", "--always"]).strip())
result_dir = os.path.join("./experiment","seed_"+str(opts["seed"]),
                          "-".join(train_opts["generator"]),
                          train_opts["assessor"],
                          "-".join(train_opts["transformer"]),
                          train_opts["version"],
                          "alpha_" + str(opts["alpha"]) + "_truncation_" + str(opts["test_truncation"]) + "_iteration_" + str(opts["checkpoint"]) + "_" + opts["mode"] + "_" + str(opts["one_way"]) + "_" + str(opts["which_way"]))

os.makedirs(result_dir, exist_ok=False)

# checkpoint_dir
checkpoint_dir = opts["checkpoint_dir"]

# Saving testing settings
opts_file = os.path.join(result_dir, "opts.json")
opts["test_version"] = test_version
with open(opts_file, 'w') as fp:
    json.dump(opts, fp)

# Some characteristics
# --------------------------------------------------------------------------------------------------------------
dim_z = {
    'biggan256': 140,
    'biggan512': 128
}.get(train_opts['generator'][0])

vocab_size = {'biggan256': 1000, 'biggan512': 1000}.get(train_opts['generator'][0])
categories_file = "./generators/categories_imagenet.txt"
categories = [x.strip() for x in open(categories_file)]

# Setting up Transformer
# --------------------------------------------------------------------------------------------------------------
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

# Setting up Generator
# --------------------------------------------------------------------------------------------------------------
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

# Setting up Assessor
# --------------------------------------------------------------------------------------------------------------
assessor_elements = getattr(assessors, train_opts['assessor'])("../../CreativeClassifier/logs/biggan+0_anc_vs_100_data96k/best.pth")
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

# Testing
# --------------------------------------------------------------------------------------------------------------
# Figure out where to resume
if opts["checkpoint"] == 0:
    checkpoint = 0
elif opts["checkpoint"] == -1:
    available_checkpoints = [x for x in os.listdir(checkpoint_dir) if x.endswith(".pth")]
    available_batch_numbers = [x.split('.')[0].split("_")[-1] for x in available_checkpoints]
    latest_number = max(available_batch_numbers)
    file_to_load = available_checkpoints[available_batch_numbers.index(latest_number)]
    transformation.load_state_dict(torch.load(os.path.join(checkpoint_dir, file_to_load)))
    checkpoint = latest_number
else:
    transformation.load_state_dict(torch.load(os.path.join(checkpoint_dir,
                                                           "pytorch_model_" + str(opts["checkpoint"]) + ".pth")))
    checkpoint = opts["checkpoint"]

# helper function


def make_image(z, y, step_size, transform, step_sizes_broadcast_y = None):
    
    if transform:
        if class_direction==1:
            z_transformed, y_transformed = transformation.transform(z, y, step_size, step_sizes_broadcast_y)
            y = y_transformed
            z = z_transformed
        else:
            if opts["one_way"] == 1:
                z_transformed = transformation.transform_test(z, y, step_size, opts["which_way"])
            else:  
                z_transformed = transformation.transform(z, y, step_size)
            
            z_transformed = z.norm() * z_transformed / z_transformed.norm()
            z = z_transformed
    
    gan_images = utils.pytorch.denorm(generator(z, y))
    gan_images_np = gan_images.permute(0, 2, 3, 1).detach().cpu().numpy()
    gan_images = input_transform(gan_images)
    gan_images = gan_images.view(-1, *gan_images.shape[-3:])
    gan_images = gan_images.to(device)

    out_scores_current = output_transform(assessor(gan_images))
    out_scores_current = out_scores_current.detach().cpu().numpy()
    if len(out_scores_current.shape) == 1:
        out_scores_current = np.expand_dims(out_scores_current, 1)

    return(gan_images_np, z, y, out_scores_current)

def read_latent_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    zs = data["latent"]
    ys = data["classes"]
    truncation = data["truncation"]
    
    ys_new = []
    ys_dict = {}
    for i in ys:
        ys_dict[i[0]] = i[1]
    for i in range(1000):
        if i in ys_dict:
            ys_new.append(ys_dict[i])
        else:
            ys_new.append(0)

    return np.reshape(np.array(zs), (1, -1)),np.reshape( np.array(ys_new), (1, -1)), truncation


# Test settings
num_samples = 1
truncation = opts["test_truncation"]
iters = 5
np.random.seed(seed=opts["seed"])
annotate = True
evaluator = MeanAssessorScoreEvaluator(alpha_values=list(np.linspace(opts["alpha"]*-5, opts["alpha"]*5, 11)))

if vocab_size == 0:
    num_categories = 1
else:
    num_categories = vocab_size-992

class_direction = opts["class_direction"]

latent_samples = glob.glob("full_general/*") 
#latent_samples = ["./artbreed_sample_1.json"]
for latent_sample in latent_samples[:1000]:

    ims = []
    outscores = []
    zs, ys, truncation = read_latent_json(latent_sample)
    latent_sample = latent_sample.split("/")[-1]
    #zs = utils.common.truncated_z_sample(num_samples, dim_z, truncation)
    #ys = np.repeat(y, num_samples)
    zs = torch.from_numpy(zs).type(torch.FloatTensor).to(device)
    ys = torch.from_numpy(ys).type(torch.FloatTensor).to(device)
    #ys = utils.pytorch.one_hot(ys, vocab_size)
    print("truncatin:",truncation )
    print("zs:", zs.shape)
    print("ys:", ys.shape)

    step_sizes = np.repeat(np.array(opts["alpha"]), num_samples * dim_z).reshape([num_samples, dim_z])
    step_sizes = torch.from_numpy(step_sizes).type(torch.FloatTensor).to(device)

    if class_direction==1:
        step_sizes_broadcast_y = np.repeat(np.array(opts["alpha"]), num_samples * vocab_size).reshape([num_samples, vocab_size])
        step_sizes_broadcast_y = torch.from_numpy(step_sizes_broadcast_y).type(torch.FloatTensor).to(device)

    feed_dicts = []
    for batch_start in range(0, num_samples, 4):
        s = slice(batch_start, min(num_samples, batch_start + 4))
        feed_dicts.append({"z": zs[s], "y": ys[s], "truncation": truncation, "step_sizes": step_sizes[s]})

    for feed_dict in feed_dicts:
        ims_batch = []
        outscores_batch = []
        z_start = feed_dict["z"]
        y_start = feed_dict["y"]
        step_sizes = feed_dict["step_sizes"]

        if opts["mode"] == "iterative":
            print("iterative")

            # original seed image
            if class_direction==0:
                x, tmp, tmpy, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], False)
            else:
                x, tmp, tmpy, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], False, step_sizes_broadcast_y)
            x = np.uint8(x)
            evaluator.update(0.0, np.mean(outscore), images=x)
            if annotate:
                ims_batch.append(utils.common.annotate_outscore(x, outscore))
            else:
                if annotate:
                    ims_batch.append(utils.common.annotate_outscore(x, outscore))
                else:
                    ims_batch.append(x)
            outscores_batch.append(outscore)

            # negative clone images
            z_next = z_start
            y_next = y_start
            step_sizes = -step_sizes
            if class_direction==1:
                step_sizes_broadcast_y = -step_sizes_broadcast_y
            for iter in range(0, iters, 1):
                feed_dict["step_sizes"] = step_sizes
                feed_dict["z"] = z_next
                feed_dict["y"] = y_next
                if class_direction==0:
                    x, tmp, tmpy, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], True)
                else:
                    x, tmp, tmpy, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], True, step_sizes_broadcast_y)
                x = np.uint8(x)
                evaluator.update(float((step_sizes * (iter + 1)).mean()), np.mean(outscore), images=x)
                z_next = tmp
                y_next = tmpy
                if annotate:
                    ims_batch.append(utils.common.annotate_outscore(x, outscore))
                else:
                    if annotate:
                        ims_batch.append(utils.common.annotate_outscore(x, outscore))
                    else:
                        ims_batch.append(x)
                outscores_batch.append(outscore)

            ims_batch.reverse()

            # positive clone images
            step_sizes = -step_sizes
            if class_direction==1:
                step_sizes_broadcast_y = -step_sizes_broadcast_y
            z_next = z_start
            y_next = y_start
            for iter in range(0, iters, 1):
                feed_dict["step_sizes"] = step_sizes
                feed_dict["z"] = z_next
                feed_dict["y"] = y_next

                if class_direction==0:
                    x, tmp, tmpy, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], True)
                else:
                    x, tmp, tmpy, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], True, step_sizes_broadcast_y)
                x = np.uint8(x)
                evaluator.update(float((step_sizes * (iter + 1)).mean()), np.mean(outscore), images=x)
                z_next = tmp
                y_next = tmpy

                if annotate:
                    ims_batch.append(utils.common.annotate_outscore(x, outscore))
                else:
                    ims_batch.append(x)
                outscores_batch.append(outscore)

        else:
            print("bigger_step")

            # original seed image
            if class_direction==0:
                x, tmp, tmpy, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], False)
            else:
                x, tmp, tmpy, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], False, step_sizes_broadcast_y)
            x = np.uint8(x)
            evaluator.update(0.0, np.mean(outscore), images=x)
            if annotate:
                ims_batch.append(utils.common.annotate_outscore(x, outscore))
            else:
                ims_batch.append(x)
            outscores_batch.append(outscore)

            # negative clone images
            step_sizes = -step_sizes
            if class_direction==1:
                step_sizes_broadcast_y = -step_sizes_broadcast_y
            for iter in range(0, iters, 1):
                feed_dict["step_sizes"] = step_sizes * (iter + 1)
                if class_direction==1:
                    step_sizes_broadcast_y_temp = step_sizes_broadcast_y * (iter + 1)
                if class_direction==0:
                    x, tmp, tmpy, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], True)
                else:
                    x, tmp, tmpy, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], True, step_sizes_broadcast_y_temp)
                x = np.uint8(x)
                evaluator.update(float(feed_dict["step_sizes"].mean()), np.mean(outscore), images=x)

                if annotate:
                    ims_batch.append(utils.common.annotate_outscore(x, outscore))
                else:
                    ims_batch.append(x)
                outscores_batch.append(outscore)

            ims_batch.reverse()
            outscores_batch.reverse()

            # positive clone images
            step_sizes = -step_sizes
            if class_direction==1:
                step_sizes_broadcast_y = -step_sizes_broadcast_y
            for iter in range(0, iters, 1):
                feed_dict["step_sizes"] = step_sizes * (iter + 1)
                if class_direction==1:
                    step_sizes_broadcast_y_temp = step_sizes_broadcast_y * (iter + 1)

                if class_direction==0:
                    x, tmp, tmpy, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], True)
                else:
                    x, tmp, tmpy, outscore = make_image(feed_dict["z"], feed_dict["y"], feed_dict["step_sizes"], True, step_sizes_broadcast_y_temp)
                x = np.uint8(x)
                evaluator.update(float(feed_dict["step_sizes"].mean()), np.mean(outscore), images=x)
                if annotate:
                    ims_batch.append(utils.common.annotate_outscore(x, outscore))
                else:
                    ims_batch.append(x)
                outscores_batch.append(outscore)

        ims_batch = [np.expand_dims(im, 0) for im in ims_batch]
        ims_batch = np.concatenate(ims_batch, axis=0)
        ims_batch = np.transpose(ims_batch, (1, 0, 2, 3, 4))
        ims.append(ims_batch)

        outscores_batch = [np.expand_dims(outscore, 0) for outscore in outscores_batch]
        outscores_batch = np.concatenate(outscores_batch, axis=0)
        #outscores_batch = np.transpose(outscores_batch, (1, 0, 2))
        outscores.append(outscores_batch)

    ims = np.concatenate(ims, axis=0)
    outscores = np.concatenate(outscores, axis=0)
    ims_final = np.reshape(ims, (ims.shape[0] * ims.shape[1], ims.shape[2], ims.shape[3], ims.shape[4]))
    I = PIL.Image.fromarray(utils.common.imgrid(ims_final, cols=iters * 2 + 1))
    if opts["one_way"] ==1:
        I.save(os.path.join(result_dir, latent_sample.replace(".json", "") + "_direction" + str(opts["which_way"]) + ".jpg"))
    else:  
        I.save(os.path.join(result_dir, latent_sample.replace(".json", "") + ".jpg"))
    print("y: ", latent_sample)
if opts["one_way"] ==1:
    evaluator.finish("assessor_score_plot.png", result_dir, opts["which_way"])
else:  
    evaluator.finish("assessor_score_plot.png", result_dir, -1)
