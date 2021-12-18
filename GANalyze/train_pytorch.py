import argparse, random
import json, cv2
import os, sys
import subprocess

from optimizers import RAdam

import numpy as np
import torch
import torch.optim as optim

import assessors
import generators
import transformations.pytorch as transformations
import utils.common
import utils.pytorch
from utils.artbreeder_util import sample_artbreeder_class



# Collect command line arguments
# --------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=0, help='which gpu to use.')
parser.add_argument('--num_samples', required=True, type=int, help='number of samples to train for')
parser.add_argument('--checkpoint_resume', type=int, default=0, help='which checkpoint to load based on batch_start. -1 for latest stored checkpoint')
parser.add_argument('--train_alpha_a', type=float, default=-0.5, help='lower limit for step sizes to use during training')
parser.add_argument('--train_alpha_b', type=float, default=0.5, help='upper limit for step sizes to use during training')
parser.add_argument('--generator', default=["biggan512", "None"], nargs=2, type=str, metavar=["name", "arguments"], help='generator function to use')
parser.add_argument('--assessor', type=str, default="creative_classifier", help='assessor function to compute the image property of interest')
parser.add_argument('--transformer', default=["OneDirection", "None"], nargs=2, type=str, metavar=["name", "arguments"], help="transformer function")

parser.add_argument('--assessor_path', type=str, required=True, help='assessor weight path')
parser.add_argument('--experiment_name', required=True, type=str, help="experiment name")
parser.add_argument('--artbreeder_class', required=True, type=int, help="0 -> use one hot class vector, 1 -> use artbreeder type sampling")
parser.add_argument('--class_direction', required=True, type=int, help="0 -> dont update class vector, 1 -> also update class vector")
parser.add_argument('--clipped_step_size', required=True, type=int, help="0 -> dont clip step sizes, 1 -> clip step sizes")
parser.add_argument('--batch_size', type=int, required=True, help='batch size')
parser.add_argument('--learning_rate', type=float, required=True, help='learning rate')
parser.add_argument('--multiway_linear', required=True, type=int, help="0 -> not multiway linear, 1 -> multiway linear")

args = parser.parse_args()
opts = vars(args)
print("Num Samples:", opts["num_samples"])

# Verify
if opts["checkpoint_resume"] != 0 and opts["checkpoint_resume"] != -1:
    assert(opts["checkpoint_resume"] % 4 == 0)  # Needs to be a multiple of the batch size

# Choose GPU
if opts["gpu_id"] != -1:
    device = torch.device("cuda:" + str(opts["gpu_id"]) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# Creating directory to store checkpoints
version = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
checkpoint_dir = os.path.join(
    "./Checkpoints/{}".format(args.experiment_name),
    "_".join(opts["generator"]),
    opts["assessor"]+"_"+opts["assessor_path"].split("/")[-1],
    "_".join(opts["transformer"]),
    "artbreeder_class:"+str(opts["artbreeder_class"]), 
    version)

print("Checkpoint: {}".format(checkpoint_dir))

if opts["checkpoint_resume"] == 0:
    os.makedirs(checkpoint_dir, exist_ok=False)

# Saving training settings
opts_file = os.path.join(checkpoint_dir, "opts.json")
opts["version"] = version
with open(opts_file, 'w') as fp:
    json.dump(opts, fp)

# Setting up file to store loss values
loss_file = os.path.join(checkpoint_dir, "losses.txt")

# Some characteristics
# --------------------------------------------------------------------------------------------------------------
dim_z = {
    'biggan256': 140,
    'biggan512': 128
}.get(opts['generator'][0])

print("Generator: {}".format(opts['generator'][0]))

vocab_size = {'biggan256': 1000, 'biggan512': 1000}.get(opts['generator'][0])

# Setting up Transformer
# --------------------------------------------------------------------------------------------------------------
transformer = opts["transformer"][0]
transformer_arguments = opts["transformer"][1]
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
generator = opts["generator"][0]
generator_arguments = opts["generator"][1]
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
#assessor_elements = getattr(assessors, opts['assessor'])(True)
assessor_elements = getattr(assessors, opts['assessor'])(args.assessor_path)
if isinstance(assessor_elements, tuple):
    assessor = assessor_elements[0]
    input_transform = assessor_elements[1]
    output_transform = assessor_elements[2]
else:
    assert False
    assessor = assessor_elements

    def input_transform(x):
        return x  # identity, no preprocessing

    def output_transform(x):
        return x  # identity, no postprocessing

if hasattr(assessor, 'parameters'):
    for p in assessor.parameters():
        p.requires_grad = False
        assessor.eval()
        assessor.to(device)

# Training
# --------------------------------------------------------------------------------------------------------------
# optimizer
#optimizer = optim.Adam(transformation.parameters(), lr=0.0002)
#optimizer = RAdam(transformation.parameters(), lr=0.0002)
optimizer = RAdam(transformation.parameters(), lr=opts["learning_rate"])
losses = utils.common.AverageMeter(name='Loss')

# figure out where to resume
if opts["checkpoint_resume"] == 0:
    checkpoint_resume = 0
elif opts["checkpoint_resume"] == -1:
    available_checkpoints = [x for x in os.listdir(checkpoint_dir) if x.endswith(".pth")]
    available_batch_numbers = [x.split('.')[0].split("_")[-1] for x in available_checkpoints]
    latest_number = max(available_batch_numbers)
    file_to_load = available_checkpoints[available_batch_numbers.index(latest_number)]
    transformation.load_state_dict(torch.load(os.path.join(checkpoint_dir, file_to_load)))
    checkpoint_resume = latest_number
else:
    transformation.load_state_dict(torch.load(os.path.join(checkpoint_dir,
                                                           "pytorch_model_{}.pth".format(opts["checkpoint_resume"]))))
    checkpoint_resume = opts["checkpoint_resume"]

#  training settings
optim_iter = 0
batch_size = int(opts["batch_size"])
train_alpha_a = opts["train_alpha_a"]
train_alpha_b = opts["train_alpha_b"]
num_samples = opts["num_samples"]

# create training set
np.random.seed(seed=0)
truncation = 1  # TODO
zs = utils.common.truncated_z_sample(num_samples, dim_z, truncation)


if args.artbreeder_class ==1:
    ys = sample_artbreeder_class(num_samples)
    print("Artbreeder type class vector sampling is used")
else:    
    ys = np.random.randint(0, vocab_size, size=zs.shape[0])
    print("One-hot class vector sampling is used")

checkpoint_loss = []

# loop over data batches
for batch_start in range(0, num_samples, batch_size):

    optimizer.zero_grad()

    # skip batches we've already done (this would happen when resuming from a checkpoint)
    if batch_start <= checkpoint_resume and checkpoint_resume != 0:
        optim_iter = optim_iter + 1
        continue

    # input batch
    s = slice(batch_start, min(num_samples, batch_start + batch_size))
    z = torch.from_numpy(zs[s]).type(torch.FloatTensor).to(device)
    
    if args.artbreeder_class == 1:
        y = torch.from_numpy(ys[s]).type(torch.FloatTensor).to(device)
    else:
        y = torch.from_numpy(ys[s]).to(device)
        #y = torch.ones_like(y)*248
            
    # ganalyze steps
    if args.artbreeder_class == 1:
        gan_images = generator(z, y)
    else:
        gan_images = generator(z, utils.pytorch.one_hot(y))
    
    gan_images = utils.pytorch.denorm(gan_images)
    gan_images_random_np = gan_images.permute(0, 2, 3, 1).detach().cpu().numpy()
    #print("debug_channel/{}.jpg".format(np.random.random()))
    #cv2.imwrite("debug_channel/swap_{}.jpg".format(np.random.random()), gan_images_np[0][:,:,[2,1,0]])
    #cv2.imwrite("debug_channel/org_{}.jpg".format(np.random.random()), gan_images_np[0])
    gan_images = input_transform(gan_images)
    gan_images = gan_images.view(-1, *gan_images.shape[-3:])
    gan_images = gan_images.to(device)
    out_scores = output_transform(assessor(gan_images)).to(device).float()

    #train_alpha_a = (batch_start/num_samples)*opts["train_alpha_a"] - 0.1
    #train_alpha_b = (batch_start/num_samples)*opts["train_alpha_b"] + 0.1

    if args.clipped_step_size:
        step_sizes = []
        out_score = out_scores.detach().cpu().numpy()
        for i in range(batch_size):
            while(1):
                temp_generated = (train_alpha_b - train_alpha_a) * random.random() + train_alpha_a
                if (out_score[i]+temp_generated) > 0 and (out_score[i]+temp_generated < 1):
                    break
            step_sizes.append(temp_generated)        
        step_sizes = np.array(step_sizes)
    else:
        step_sizes = (train_alpha_b - train_alpha_a) * \
            np.random.random(size=(batch_size)) + train_alpha_a  # sample step_sizes
    

    step_sizes_broadcast = np.repeat(step_sizes, dim_z).reshape([batch_size, dim_z])
    step_sizes_broadcast = torch.from_numpy(step_sizes_broadcast).type(torch.FloatTensor).to(device)

    if args.class_direction == 1:
        step_sizes_broadcast_y = np.repeat(step_sizes, vocab_size).reshape([batch_size, vocab_size])
        step_sizes_broadcast_y = torch.from_numpy(step_sizes_broadcast_y).type(torch.FloatTensor).to(device)
        
    target_scores = out_scores + torch.from_numpy(step_sizes).to(device).float()
    
    # assert not(args.artbreeder_class == 0 and args.class_direction == 1) 

    if args.artbreeder_class == 1:
        if args.class_direction == 1:
            z_transformed, y_transformed = transformation.transform(z, y, step_sizes_broadcast, step_sizes_broadcast_y)
            gan_images_transformed = generator(z_transformed, y_transformed)
        else:
            z_transformed = transformation.transform(z, y, step_sizes_broadcast)
            gan_images_transformed = generator(z_transformed, y)

    else:
        if args.class_direction == 0:
            z_transformed = transformation.transform(z, utils.pytorch.one_hot(y), step_sizes_broadcast)
            gan_images_transformed = generator(z_transformed, utils.pytorch.one_hot(y))
        else:
            z_transformed, y_transformed = transformation.transform(z, utils.pytorch.one_hot(y), step_sizes_broadcast, step_sizes_broadcast_y)
            gan_images_transformed = generator(z_transformed, y_transformed)

    gan_images_transformed = utils.pytorch.denorm(gan_images_transformed)
    gan_images_transformed_np = gan_images_transformed.permute(0, 2, 3, 1).detach().cpu().numpy()

    gan_images_transformed = input_transform(gan_images_transformed)
    gan_images_transformed = gan_images_transformed.view(-1, *gan_images_transformed.shape[-3:])
    gan_images_transformed = gan_images_transformed.to(device)
    out_scores_transformed = output_transform(assessor(gan_images_transformed)).to(device).float()
    
    # out_scores_transformed[(out_scores>1) | (out_scores<0)] = target_scores[(out_scores>1) | (out_scores<0)]
    # out_scores_transformed[(out_scores_transformed>1) | (out_scores_transformed<0)] = target_scores[(out_scores_transformed>1) | (out_scores_transformed<0)]

    # compute loss
    if args.multiway_linear == 1:
        loss = transformation.compute_loss(out_scores_transformed, target_scores, batch_start, loss_file, z)
    else:
        loss = transformation.compute_loss(out_scores_transformed, target_scores, batch_start, loss_file)

    checkpoint_loss.append(loss.item())
    # backwards
    loss.backward()

    #torch.nn.utils.clip_grad_norm_(transformation.parameters(), 1)
    optimizer.step()

    # print loss
    losses.update(loss.item(), batch_size)
    
    if batch_start == 0:
        assert not os.path.exists("train_logs/{}.log".format(args.experiment_name))
    
    checkpoint_avg_loss = np.mean(checkpoint_loss)

    with open("train_logs/{}.log".format(args.experiment_name),"a+") as file:
        file.write(f'[{batch_start}/{num_samples}] {losses} - {checkpoint_avg_loss}\n')
        file.write("out: "+str(list(out_scores.detach().cpu().numpy()))+" - target: "+str(list(target_scores.detach().cpu().numpy()))+" - after: "+str(list(out_scores_transformed.detach().cpu().numpy()))+"\n")#+" - grad: "+ str(transformation.w.grad.mean().item()) +"\n")

    print(f'[{batch_start}/{num_samples}] {losses} - {checkpoint_avg_loss}')

    if optim_iter % 250 == 0:
        checkpoint_loss = []
        print("saving checkpoint")
        torch.save(transformation.state_dict(), os.path.join(checkpoint_dir, "pytorch_model_{}.pth".format(batch_start)))
        
        # debug image write
        os.makedirs("train_debug_images/{}".format(args.experiment_name), exist_ok=True)
        random_images = []
        transformed_images = []
        random_scores = out_scores.detach().cpu().numpy()
        transformed_scores = out_scores_transformed.detach().cpu().numpy()
        for i, img in enumerate(gan_images_random_np):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.putText(img, "{:.3f}".format(random_scores[i]), (50,50), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 255, 0) , 2, cv2.LINE_AA)
            img = cv2.putText(img, "{:.3f}".format(step_sizes[i]), (50,450), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 255, 0) , 2, cv2.LINE_AA)
            random_images.append(img)

        for i, img in enumerate(gan_images_transformed_np):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.putText(img, "{:.3f}".format(transformed_scores[i]), (50,50), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (255, 255, 0) , 2, cv2.LINE_AA)
            transformed_images.append(img)
        
        cv2.imwrite("train_debug_images/{}/{}_{:.3f}.jpg".format(args.experiment_name, batch_start, checkpoint_avg_loss), np.hstack([np.vstack(random_images), np.vstack(transformed_images)]))
    
    optim_iter = optim_iter + 1

torch.save(transformation.state_dict(), os.path.join(checkpoint_dir, "pytorch_model_{}.pth".format(opts["num_samples"])))
