"""
train.py
"""
import os
import argparse

import torch
import torch.optim as optim
from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
from pyhandle.net.intermediate import IntermediateNetwork

from net.ssd import SSD300
from net.ssd import MultiBoxLoss
from dataset.obj_dataloader import ObjTorchLoader


DEVICE = None

def train_for_one_step(model, criterion,
                       optimizer, loss_container,
                       inputs, boxes, labels):
    # zero the parameter gradients
    # optimizer.zero_grad()
    for param in model.parameters():
        param.grad = None

    # forward + backward + optimize
    pred_locs, pred_cls_prob = model(inputs.to(DEVICE))

    num_obj = len(boxes)
    for n in range(num_obj):
        boxes[n] = boxes[n].to(DEVICE)
        labels[n] = labels[n].to(DEVICE)
    loss = criterion(pred_locs.to(DEVICE), pred_cls_prob.to(DEVICE),
                     boxes, labels)
    loss.backward()
    optimizer.step()
    loss_container[0] = loss.item()

    del loss


def train_for_one_epoch(model, criterion, optimizer,
                        dataset, epoch, step_per_epoch,
                        writer=None):
    step = 0
    losses = [0.]
    step_to_draw_loss = 10
    for images, boxes, labels in dataset.train_loader:
        train_for_one_step(model, criterion,
                           optimizer, losses,
                           images, boxes, labels)

        if writer is not None and step % step_to_draw_loss == 0:
            writer.add_scalar("Loss/train", losses[0], step + epoch * step_per_epoch)

        step += 1


def train_loop(training_setup, epoch):
    model = training_setup['model']
    criterion = training_setup['criterion']
    optimizer = training_setup['optimizer']
    dataset = training_setup['dataset']
    writer = training_setup['writer']

    step_per_epoch = len(dataset.train_loader)
    for e in range(epoch):
        model.train()
        train_for_one_epoch(model, criterion, optimizer, dataset, e, step_per_epoch, writer=writer)

        # eval_result = eval_for_one_epoch(training_setup['dataset'], training_setup['model'])


def setting_up(args):
    setup = {}
    backbone = IntermediateNetwork('resnet50', [5, 6]).to(DEVICE)
    setup['model'] = SSD300(backbone, args.num_classes).to(DEVICE)
    setup['preprocessor'] = transforms.Compose([transforms.Resize((300, 300)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])
    setup['dataset'] = ObjTorchLoader(args.dataset_name,
                                      transform=setup['preprocessor'],
                                      collate_fn_name=args.dataset_name,
                                      train_batch_size=args.train_batch_size,
                                      test_batch_size=args.test_batch_size)
    prior_boxes = setup['model'].priors_cxcy
    setup['criterion'] = MultiBoxLoss(prior_boxes)
    setup['optimizer'] = optim.SGD(setup['model'].parameters(),
                                   lr=args.lr)
    setup['writer'] = None
    setup['save_model_path'] = args.save_model_path

    return setup


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='resnet18')
    parser.add_argument('--dataset_name', default='coco')
    parser.add_argument('--num_classes', type=int, default=91)
    parser.add_argument('--dataset_path', default=None)
    parser.add_argument('--training_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--save_model_path', type=str, default=None)
    parser.add_argument('--if_eval', type=bool, default=False)
    return parser.parse_args()


def train(arguments):
    setup = setting_up(arguments)
    train_loop(setup, epoch=arguments.training_epoch)


if __name__ == '__main__':
    args = get_arguments()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.backends.cudnn.benchmark = True
    train(args)
