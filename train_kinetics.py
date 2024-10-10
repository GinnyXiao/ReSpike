import argparse
import shutil
import os
import time
import torch
import logging as logger
from tqdm import tqdm
import torch.nn as nn
from torch import autocast
from torch.cuda.amp import GradScaler
from utils import seed_all
from dataset.video_dataset import KineticsDataset
from dataset.transforms import *
from torchvision import transforms
from model.attention_snn import CrossAttenFusion

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='PyTorch Temporal Efficient Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--dset', default='kinetics400', type=str, metavar='N', choices=['kinetics400'],
                    help='dataset')
parser.add_argument('--model', default='resnet18', type=str, metavar='N', choices=['resnet18', 'resnet50'],
                    help='ANN architecture')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--seed', default=1001, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-T', '--time', default=16, type=int, metavar='N',
                    help='snn simulation time (default: 16)')
parser.add_argument('--stride', default=4, type=int, metavar='N',
                    help='key frame stride')
parser.add_argument('--amp', action='store_false',
                    help='if use amp training.')
parser.add_argument('--downsample', action='store_true',
                    help='if use downsample for cross-atten calculation.')
args = parser.parse_args()


def train(model, device, train_loader, criterion, optimizer, epoch, scaler, args):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    s_time = time.time()
    for  (images, labels) in tqdm(train_loader):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)

        if args.amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
                scaler.scale(loss.mean()).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.mean().backward()
            optimizer.step()

        running_loss += loss.item()
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    e_time = time.time()
    return running_loss / M, 100 * correct / total, (e_time-s_time)/60


@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for  (inputs, targets) in tqdm(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())

    final_acc = 100 * correct / total
    return final_acc


if __name__ == '__main__':

    seed_all(args.seed)
    
    # ----------------------------------- dataset config -----------------------------------
    input_size = 224
    clip_len = args.time

    data_root_train = "/datasets/Kinetics-400/videos_train/"
    ann_file_train = "/datasets/Kinetics-400/kinetics400_train_list_videos.txt"
    data_root_val = "/datasets/Kinetics-400/videos_val/"
    ann_file_val = "/datasets/Kinetics-400/kinetics400_val_list_videos.txt"

    train_augmentation = transforms.Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                                            GroupRandomHorizontalFlip(is_flow=False)])
    input_mean=[0.485, 0.456, 0.406]
    input_std=[0.229, 0.224, 0.225]
    normalize = GroupNormalize(input_mean, input_std)
    scale_size = input_size * 256 // 224

    print("Creating training datasets...")
    train_dataset = KineticsDataset(dataset="kinetics",
                                    root_path=data_root_train,
                                    list_file=ann_file_train,
                                    num_segments=16,
                                    new_length=1,
                                    transform=transforms.Compose([train_augmentation,
                                        Stack(roll=False),
                                        ToTorchFormatTensor(div=True),
                                        normalize,]),
                                    random_shift=True, test_mode=False, )  # True for test mode

    print("Creating validation datasets...")

    val_dataset = KineticsDataset(dataset="kinetics",
                                  root_path=data_root_val,
                                  list_file=ann_file_val,
                                  num_segments=16,
                                  new_length=1,
                                  transform=transforms.Compose([
                                    GroupScale(int(scale_size)),
                                    GroupCenterCrop(input_size),
                                    Stack(roll=False),
                                    ToTorchFormatTensor(div=True),
                                    normalize,]),
                                  random_shift=False, test_mode=True, )  # True for test mode

    print(len(train_dataset))
    print(len(val_dataset))

    num_classes = 400

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, )


    # ----------------------------------- model config ---------------------------------------

    model = CrossAttenFusion(backbone=args.model, key_frame_stride=args.stride, width_mult=4, num_classes=num_classes, downsample=args.downsample)

    model.cuda()
    device = next(model.parameters()).device

    # ----------------------------------- optimizer config -----------------------------------
    scaler = GradScaler() if args.amp else None

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    # ----------------------------------- training config -----------------------------------
    best_acc = 0
    best_epoch = 0

    save_params = (args.dset, args.model, str(args.stride), str(args.lr))
    save_names = '-'.join(save_params) + 'shuffle-run1.pth'
    print(save_names)

    print('start training!')
    for epoch in range(args.epochs):

        loss, acc, t_diff = train(model, device, train_loader, criterion, optimizer, epoch, scaler, args)
        print('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f},\t time elapsed: {}'.format(epoch, args.epochs, loss, acc,
                                                                                    t_diff))
        scheduler.step()
        facc = test(model, test_loader, device)
        print('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch, args.epochs, facc))

        if best_acc < facc:
            best_acc = facc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_names)
            print("saving model checkpoint...")
        print('Best Test acc={:.3f}'.format(best_acc))
