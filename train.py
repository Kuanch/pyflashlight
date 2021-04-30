import argparse

import torch
import torch.optim as optim
from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter
from pyhandle.net.intermediate import IntermediateNetwork

from net.ssd import SSD300
from net.ssd import MultiBoxLoss
from dataset.obj_dataloader import ObjTorchLoader


DEVICE = 'cuda:0'


def collate_fn_coco(batch):
    images, annos = tuple(zip(*batch))
    t_images = torch.empty((0, 3, 300, 300))
    b_bboxes = []
    b_labels = []
    for i, image in enumerate(images):
        labels = []
        bboxes = []
        r_width = 1 / image.shape[0]
        r_height = 1 / image.shape[1]
        t_image = torch.unsqueeze(image, dim=0)
        t_images = torch.cat((t_images, t_image))
        for anno in annos[i]:
            bbox = [anno['bbox'][0] * r_width, anno['bbox'][1] * r_height,
                    (anno['bbox'][0] + anno['bbox'][2]) * r_width, (anno['bbox'][1] + anno['bbox'][3]) * r_height]
            bboxes.append(bbox)
            labels.append(anno['category_id'])
        b_bboxes.append(torch.as_tensor(bboxes))
        b_labels.append(torch.as_tensor(labels))

    return t_images, b_bboxes, b_labels


def train_for_one_step(model, criterion,
                       optimizer, loss_container,
                       inputs, boxes, labels):
    # zero the parameter gradients
    # optimizer.zero_grad()
    for param in model.parameters():
        param.grad = None

    # forward + backward + optimize
    pred_locs, pred_cls_prob = model(inputs.to(DEVICE))

    for b in range(len(boxes)):
        boxes[b].to(DEVICE)
        labels[b].to(DEVICE)

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
    optim = training_setup['optimizer']
    dataset = training_setup['dataset']
    writer = training_setup['writer']

    step_per_epoch = len(dataset.train_loader)
    for e in range(epoch):
        model.train()
        train_for_one_epoch(model, criterion, optim, dataset, e, step_per_epoch, writer=writer)

        # eval_result = eval_for_one_epoch(training_setup['dataset'], training_setup['model'])


def training_setup(args):
    training_setup = {}
    backbone = IntermediateNetwork('resnet50', [5, 6]).to(DEVICE)
    training_setup['model'] = SSD300(backbone, args.num_classes).to(DEVICE)
    training_setup['preprocessor'] = transforms.Compose([transforms.Resize((300, 300)),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])])
    collate_fn = collate_fn_coco
    training_setup['dataset'] = ObjTorchLoader(args.dataset_name,
                                               transform=training_setup['preprocessor'],
                                               collate_fn=collate_fn,
                                               train_batch_size=args.train_batch_size,
                                               test_batch_size=args.test_batch_size)
    prior_boxes = training_setup['model'].priors_cxcy.to(DEVICE)
    training_setup['criterion'] = MultiBoxLoss(prior_boxes)
    training_setup['optimizer'] = optim.SGD(training_setup['model'].parameters(),
                                            lr=args.lr)
    training_setup['writer'] = None
    training_setup['save_model_path'] = args.save_model_path

    return training_setup


def get_argments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='resnet18')
    parser.add_argument('--dataset_name', default='coco')
    parser.add_argument('--num_classes', type=int, default=91)
    parser.add_argument('--dataset_path', default=None)
    parser.add_argument('--training_epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--save_model_path', type=str, default=None)
    parser.add_argument('--if_eval', type=bool, default=False)
    return parser.parse_args()


def train(args):
    setup = training_setup(args)
    train_loop(setup, epoch=args.training_epoch)


if __name__ == '__main__':
    args = get_argments()
    torch.backends.cudnn.benchmark = True
    train(args)

