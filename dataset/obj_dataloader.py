import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class ObjTorchLoader(object):
    def __init__(self, dataset_name,
                 transform=None,
                 collate_fn=None,
                 train_batch_size=16,
                 test_batch_size=8,
                 data_path=None):
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        coco_root = '/home/sixigma/workplace/meow/coco_data/'
        if hasattr(torchvision.datasets, dataset_name):
            train_set = torchvision.datasets.coco.CocoDetection(root=coco_root + 'val/val2017/',
                                                      annFile=coco_root + 'annotations/instances_val2017.json',
                                              transform=transform)
            self.train_loader = DataLoader(train_set,
                                           batch_size=train_batch_size,
                                           shuffle=True,
                                           num_workers=1,
                                           collate_fn=collate_fn)

            test_set = torchvision.datasets.coco.CocoDetection(root=coco_root + 'test/test2017/',
                                                      annFile=coco_root + 'annotations/image_info_test2017.json',
                                             transform=transform)
            self.test_loader = DataLoader(test_set,
                                          batch_size=test_batch_size,
                                          shuffle=True,
                                          num_workers=1,
                                          collate_fn=collate_fn)
        else:
            raise ImportError('unknown dataset {}, only torchvision seems not to support'.format(dataset_name))

    def read_train(self):
        images, boxes, labels = next(iter(self.train_loader))
        return images.cuda(), boxes.cuda(), labels.cuda()

    def read_test(self):
        images, boxes, labels = next(iter(self.test_loader))
        return images.cuda(), boxes.cuda(), labels.cuda()
