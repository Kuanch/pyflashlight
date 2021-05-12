import torch


def collate_fn_coco(batch):
    images, annos = tuple(zip(*batch))
    t_images = torch.empty((0, 3, 300, 300))
    num_objs = torch.empty((len(images)), dtype=torch.int64)
    b_bboxes = torch.empty((0, 4), dtype=torch.float32)
    b_labels = torch.empty((0), dtype=torch.int64)
    for i, image in enumerate(images):
        r_width = 1 / image.shape[0]
        r_height = 1 / image.shape[1]
        t_image = torch.unsqueeze(image, dim=0)
        t_images = torch.cat((t_images, t_image))
        num_objs[i] = len(annos[i])
        boxes = torch.empty((len(annos[i]), 4), dtype=torch.float32)
        labels = torch.empty((len(annos[i])), dtype=torch.int64)
        for obj, anno in enumerate(annos[i]):
            boxes[obj][0] = anno['bbox'][0] * r_width
            boxes[obj][1] = anno['bbox'][1] * r_height
            boxes[obj][2] = (anno['bbox'][0] + anno['bbox'][2]) * r_width
            boxes[obj][3] = (anno['bbox'][1] + anno['bbox'][3]) * r_height
            labels[obj] = anno['category_id']
        b_bboxes = torch.cat((b_bboxes, boxes), axis=0)
        b_labels = torch.cat((b_labels, labels), axis=0)

    return t_images, b_bboxes, b_labels, num_objs

COLLATE_FN = {'coco': collate_fn_coco}

def get_collate_fn(name):
    if name not in COLLATE_FN:
        print('{} not supported, \
              add collate function in collation.py. \
              collate_fn=None is returned.'.format(name))
        return None
    
    return COLLATE_FN[name]
