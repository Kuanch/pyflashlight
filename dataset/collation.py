import torch


def collate_fn_coco(batch):
    images, annos = tuple(zip(*batch))
    t_images = torch.empty((0, 3, 300, 300))
    b_bboxes = []
    b_labels = []
    for i, image in enumerate(images):
        r_width = 1 / image.shape[0]
        r_height = 1 / image.shape[1]
        t_image = torch.unsqueeze(image, dim=0)
        t_images = torch.cat((t_images, t_image))
        boxes = torch.empty((len(annos[i]), 4), dtype=torch.float32)
        labels = torch.empty((len(annos[i])), dtype=torch.int64)
        for num_obj, anno in enumerate(annos[i]):
            boxes[num_obj][0] = anno['bbox'][0] * r_width
            boxes[num_obj][1] = anno['bbox'][1] * r_height
            boxes[num_obj][2] = (anno['bbox'][0] + anno['bbox'][2]) * r_width
            boxes[num_obj][3] = (anno['bbox'][1] + anno['bbox'][3]) * r_height
            labels[num_obj] = anno['category_id']
        b_bboxes.append(boxes)
        b_labels.append(labels)

    return t_images, b_bboxes, b_labels

COLLATE_FN = {'coco': collate_fn_coco}

def get_collate_fn(name):
    if name not in COLLATE_FN:
        print('{} not supported, \
              add collate function in collation.py. \
              collate_fn=None is returned.'.format(name))
        return None
    
    return COLLATE_FN[name]
