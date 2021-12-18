import os, argparse, shutil, sys, time
from datetime import datetime

import torch
from torch.utils.data.dataloader import DataLoader
#from torch.utils.tensorboard import SummaryWriter

from efficientnet_pytorch import EfficientNet

from core.dataset import *
from core.optimizers import RAdam
from core.utils import time_to_str, read_config
from core.losses import LabelSmoothingCrossEntropy

torch.backends.cudnn.benchmark = True


def main(config):
    cfg = read_config(config)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_path = cfg.data.train_dir
    val_path = cfg.data.val_dir

    train_dataset = CreativeDataset(train_path, debug=cfg.debug, data_cfg=cfg.data)
    val_dataset = CreativeDataset(val_path, debug=cfg.debug, data_cfg=cfg.data, aug=False, is_train=False)
    
    train_cfg = cfg.train

    # Data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size,
                            shuffle=True, pin_memory=True, \
                            num_workers=train_cfg.num_workers)

    val_loader = DataLoader(dataset=val_dataset, batch_size=train_cfg.batch_size,
                            shuffle=False, pin_memory=True, \
                            num_workers=train_cfg.num_workers)

    train_name = cfg.train_name

    experiment_path = os.path.join(cfg.log_folder, train_name)
    #writer = SummaryWriter(log_dir=experiment_path)

    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
    
    pretrained_path = train_cfg.pretrained
    shutil.copy2("./train.py", experiment_path)
    shutil.copy2(config, experiment_path)

    # Model initialization
    if os.path.exists(pretrained_path) and pretrained_path != "":
        model = torch.load(pretrained_path)
    else:
        model = EfficientNet.from_pretrained(train_cfg.model, num_classes=2 if cfg.data.task_type == "classification" else 1).to(device)
        start_epoch = 0
        if cfg.data.task_type == "classification":
            best_metric = 0.0  # accuracy or f1
        elif cfg.data.task_type == "regression":
            best_metric = 10e5 # loss or distance

    # classification losses
    if cfg.data.task_type == "classification":
        if cfg.train.loss == "CrossEntropy":
            criterion = torch.nn.CrossEntropyLoss()
        elif cfg.train.loss == "LabelSmoothingCrossEntropy":
            criterion = LabelSmoothingCrossEntropy()
        else:
            assert False, "wrong loss function"

    elif cfg.data.task_type == "regression":
        if cfg.train.loss == "MSELoss":
            criterion = torch.nn.MSELoss()
        elif cfg.train.loss == "L1Loss":
            criterion = torch.nn.L1Loss()
        else:
            assert False, "wrong loss function"

    if cfg.train.optimizer == "RAdam":
        optimizer = RAdam(model.parameters(), lr=train_cfg.learning_rate,\
                        weight_decay=train_cfg.weight_decay)
    elif cfg.train.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate,\
                        weight_decay=train_cfg.weight_decay)
    else:
        assert False, "wrong optimizer"

    # Training
    step = 0

    training_start = time.time()
    log_interval = len(train_loader) // 5

    for epoch in range(start_epoch, train_cfg.num_epoch):
        
        loss_list = []
        model.train()
        correct = 0

        for i, (images, labels) in enumerate(train_loader):
            step = (epoch * len(train_loader) + i + 1)
            images = images.to(device)
            labels = labels.to(device).float()

            if cfg.data.task_type == "classification":
                labels = labels.long()
            
            # Forward pass
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            avg_loss = np.mean(loss_list)
            # writer.add_scalar("loss/train", loss.item(), step)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end = time.time()
            
            if (step) % log_interval == 0:
                print()
            sys.stdout.write(
                '\rStep [{}/{}], Loss: {}, Avg Loss: {}, Elapsed Time: {}'\
            .format(step, train_cfg.num_epoch * len(train_loader), loss.item(), avg_loss,
                            time_to_str(end - training_start)))
            sys.stdout.flush()
            

        if epoch % train_cfg.validation_interval == 0:
            
            val_loss_list = []
            correct = 0
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            model.eval()

            with torch.no_grad():
                for _, (images, labels) in enumerate(val_loader):

                    images = images.to(device)
                    labels = labels.to(device).float()
                    if cfg.data.task_type == "classification":
                        labels = labels.long()
                    
                    outputs = model(images)
                    val_loss_list.append(criterion(outputs, labels).item())

                    if cfg.data.task_type == "classification":
                        y_pred = outputs.argmax(axis=1)
                        y_true = labels
                        correct += (y_pred == y_true).sum().item()
                        tp += (y_true * y_pred).sum().to(torch.float32)
                        tn += ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
                        fp += ((1 - y_true) * y_pred).sum().to(torch.float32)
                        fn += (y_true * (1 - y_pred)).sum().to(torch.float32)

                    elif cfg.data.task_type == "regression":
                        assert False, "need to be modified"
                        outputs_np = outputs.detach().cpu().numpy()
                        labels_np = labels.detach().cpu().numpy()
                        outputs_np[outputs_np>0.5] = 1.0
                        outputs_np[outputs_np<=0.5] = 0.0
                        correct += (outputs_np == labels_np).sum().item()

            avg_val_loss = np.mean(val_loss_list)
            if cfg.data.task_type == "classification":
                val_acc = correct/len(val_dataset)
                epsilon = 1e-7
                precision = tp / (tp + fp + epsilon)
                recall = tp / (tp + fn + epsilon)
                val_f1 = 2* (precision*recall) / (precision + recall + epsilon)
                val_metric = val_f1
                print("\nStep: {}, Val Loss: {}, Val Acc: {}, Val P:{}, Val R:{} Val F1: {}"\
                                .format(step, avg_val_loss, val_acc, precision, recall, val_f1))
            else:
                val_metric = avg_val_loss
                print("\nStep: {}, Val Loss: {}".format(step, avg_val_loss))

            if val_metric > best_metric:
                best_metric = val_metric
                torch.save(model, os.path.join(experiment_path, "best.pth"))

                print("Saving model on step: {} with metric of {}".format(step, best_metric))
                
                log_file = open("{}/{}.txt".format(experiment_path, train_name), "a")

                if cfg.data.task_type == "regression":
                    log_file.write(
                        "Step: {}, Val loss: {}\n".format(step, avg_val_loss))
                else:
                    log_file.write(
                        "Step: {}, Val Loss: {}, Val Acc: {}, Val P:{}, Val R:{}, Val F1: {}\n".\
                            format(step, avg_val_loss, val_acc, precision, recall, val_f1)
                    )

                log_file.write("Saving model on step: {}\n".format(step))
                log_file.close()
            else:
                # writer.add_scalar("loss/val", avg_val_loss, epoch + 1)
                #writer.add_scalar("acc/val", val_acc, epoch + 1)
                # writer.flush()
                pass
        epoch_end = time.time()
        remaining = time_to_str((train_cfg.num_epoch - epoch - 1) * ((epoch_end - training_start) / (epoch + 1)))
        print("Epoch {} is Done! Remaining Time: {}".format(epoch + 1, remaining))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.yaml", help="Train config path")
    args = parser.parse_args()
    main(args.config)
