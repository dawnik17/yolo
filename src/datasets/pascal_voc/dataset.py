"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=[13, 26, 52],
        transform=None,
        load_mosaic=True
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.load_mosaic = load_mosaic
        
        # shape: [number of states, number of anchors, 2]
        self.anchors = torch.tensor(anchors)
        self.num_anchors_per_scale = self.anchors.shape[1]

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def iou(box, anchors):
        """
        box:
            tensor shape: [2]
        anchors:
            tensor shape: [number of states, number of anchors, 2]
            
        * 2 above is for width and height
        """

        intersection = torch.prod(torch.min(box, anchors), dim=-1)
        union = torch.prod(box) + torch.prod(anchors, dim=-1) - intersection
        return intersection / union

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])

        # bboxes = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1)
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        """
        Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        6 = [objectness, cx, cy, w, h, class]
        """
        targets = [torch.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.S]

        for bbox in bboxes:
            iou = self.iou(torch.tensor(bbox[2:4]), self.anchors)

            idx = torch.argsort(iou, descending=True, dim=-1)
            idx = idx[:, 0].tolist()

            dimensions, class_ = np.array(bbox[:-1]), bbox[-1]#+1

            for scale_idx, anchor_id in enumerate(idx):
                scale_dim = self.S[scale_idx]
                scale_cx, scale_cy, scale_w, scale_h = dimensions * scale_dim
                
                row, col = int(scale_cy), int(scale_cx)

                # fill values
                scale_cx = scale_cx - col
                scale_cy = scale_cy - row

                box_target = torch.tensor(
                    [1, scale_cx, scale_cy, scale_w, scale_h, class_]
                )

                targets[scale_idx][anchor_id, row, col] = box_target

        return image, targets
    

if __name__ == "__main__":
    from src.run.yolov3 import config

    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset = YOLODataset(
        config.DATASET + "/2examples.csv",
        transform=config.train_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )