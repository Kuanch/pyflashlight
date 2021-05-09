import os

import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset.collation import get_collate_fn


class ObjTorchLoader(object):
    def __init__(self, dataset_name,
                 transform=None,
                 collate_fn_name=None,
                 train_batch_size=16,
                 test_batch_size=8,
                 data_path=None):
        if dataset_name == 'coco':
            collate_fn = get_collate_fn(collate_fn_name)
            if transform is None:
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.471, 0.448, 0.408),
                                                                     (0.234, 0.239, 0.242))])
            self.train_loader, self.test_loader = self.__get_coco(transform,
                                                                  train_batch_size,
                                                                  test_batch_size,
                                                                  collate_fn)
        elif hasattr(torchvision.datasets, dataset_name):
            collate_fn = get_collate_fn(collate_fn_name)
            if transform is None:
                # ImageNet mean and std
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406),
                                                                     (0.229, 0.224, 0.225))])

            train_set = getattr(torchvision.datasets, dataset_name)(root='./data',
                                                                    train=True,
                                                                    download=True,
                                                                    transform=transform)
            
            self.train_loader = DataLoader(train_set,
                                           batch_size=train_batch_size,
                                           shuffle=True,
                                           num_workers=1,
                                           collate_fn=collate_fn)

            test_set = getattr(torchvision.datasets, dataset_name)(root='./data',
                                                                    train=False,
                                                                    download=True,
                                                                    transform=transform)

            self.test_loader = DataLoader(test_set,
                                          batch_size=test_batch_size,
                                          shuffle=True,
                                          num_workers=1,
                                          collate_fn=collate_fn)
            
        else:
            raise ImportError('unknown dataset {}, \
                              only torchvision datasets are supported'.format(dataset_name))
        
    def __get_coco(self, transform, train_batch_size,
                   test_batch_size, collate_fn):
        coco_root = os.environ['COCO_ROOT']
        train_set = torchvision.datasets.coco.CocoDetection(root=coco_root + 'val/val2017/',
                                                            annFile=coco_root + 'annotations/instances_val2017.json',
                                                            transform=transform)
        train_loader = DataLoader(train_set,
                                  batch_size=train_batch_size,
                                  shuffle=True,
                                  num_workers=1,
                                  collate_fn=collate_fn)

        test_set = torchvision.datasets.coco.CocoDetection(root=coco_root + 'test/test2017/',
                                                           annFile=coco_root + 'annotations/image_info_test2017.json',
                                                           transform=transform)
        test_loader = DataLoader(test_set,
                                 batch_size=test_batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 collate_fn=collate_fn)
        
        return train_loader, test_loader

    def read_train(self):
        images, boxes, labels = next(iter(self.train_loader))
        return images.cuda(), boxes.cuda(), labels.cuda()

    def read_test(self):
        images, boxes, labels = next(iter(self.test_loader))
        return images.cuda(), boxes.cuda(), labels.cuda()
