"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class YoloLoss(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 5  # 1.5
        self.lambda_noobj = 2
        self.lambda_obj = 1
        self.lambda_box = 2

        self.nclasses = nclasses

    # intersection over union
    @staticmethod
    def iou(box1, box2):
        """
        boxi shape = [any shape, 4] i.e [4] or [3,4] or [2,3,4] etc.

        * 4 = [x, y, w, h]

        output shape = [batch]
        """
        # box1 x1, x2
        box1_x1 = box1[..., 0] - box1[..., 2] / 2
        box1_x2 = box1[..., 0] + box1[..., 2] / 2

        # box2 x1, x2
        box2_x1 = box2[..., 0] - box2[..., 2] / 2
        box2_x2 = box2[..., 0] + box2[..., 2] / 2

        # the width of intersection (x)
        x1 = torch.max(box1_x1, box2_x1)
        x2 = torch.min(box1_x2, box2_x2)

        x = (x2 - x1).clamp(0)

        # box1 y1, y2
        box1_y1 = box1[..., 1] - box1[..., 3] / 2
        box1_y2 = box1[..., 1] + box1[..., 3] / 2

        # box2 y1, y2
        box2_y1 = box2[..., 1] - box2[..., 3] / 2
        box2_y2 = box2[..., 1] + box2[..., 3] / 2

        # the height of intersection (y)
        y1 = torch.max(box1_y1, box2_y1)
        y2 = torch.min(box1_y2, box2_y2)

        y = (y2 - y1).clamp(0)

        # intersection
        intersection = x * y

        # union
        area_box1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        area_box2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

        union = area_box1 + area_box2 - intersection + 1e-6

        return intersection / union

    def forward(self, predictions, target, anchors):
        """
        predictions: [batch, 3, 13, 13, 25] where 25 = [objectness, cx, cy, w, h] + 20 classes
        target: [batch, 3, 13, 13, 6] where 6 = [objectness, cx, cy, w, h, true class]
        anchors: [3, 2]

        * 13 is S
        * 3 is number of anchors
        """
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        """
        both - no object and object loss
        uncomment noobj above and below for only no_object_loss)
        
        predictions shape: [batch, 3, 13, 13]
        target shape: [batch, 3, 13, 13]
        """
        no_object_loss = self.bce(predictions[..., 0][noobj], target[..., 0][noobj])

        """
        object loss
        
        predictions[..., 0][obj] shape: [total_object_in_batch]
        target[..., 0][obj] shape: [total_object_in_batch]
        """
        object_loss = self.bce(predictions[..., 0][obj], target[..., 0][obj])

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat(
            [
                self.sigmoid(predictions[..., 1:3]),
                torch.exp(predictions[..., 3:5]) * anchors,
            ],
            dim=-1,
        )

        ious = self.iou(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss += self.mse(
            self.sigmoid(predictions[..., 0][obj]), ious * target[..., 0][obj]
        )

        """
        coordinate loss or box loss
        
        predictions[..., 1:5][obj] shape: [total_obj_in_batch, 4]
        target[..., 1:5][obj] shape: [total_obj_in_batch, 4]
        """
        # x, y coordinates
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])

        # width, height coordinates
        target[..., 3:5] = torch.log((1e-16 + target[..., 3:5] / anchors))
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        """
        classification loss : cross entropy
        
        predictions[..., 5:][obj] shape: [total_obj_in_batch, nclasses]
        target[..., 5][obj].long() shape: [total_obj_in_batch]
        """
        class_loss = self.entropy(
            (predictions[..., 5:][obj]),
            (target[..., 5][obj].long()),
        )

        """
        classification loss : binary cross entropy
        
        This is my innovation: could be wrong
        Train and test without it as well.        
        """
        binary_class_loss = self.bce(
            predictions[..., 5:][obj],
            F.one_hot(target[..., 5][obj].long(), num_classes=self.nclasses).float(),
        )

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
            + self.lambda_class * binary_class_loss
        )


if __name__ == "__main__":
    from src.run.yolov3 import config
    from src.datasets.pascal_voc import YOLODataset

    S = 13
    yl = YoloLoss(nclasses=20)

    predictions = torch.rand((20, 3, S, S, 25))

    # build target
    IMAGE_SIZE = config.IMAGE_SIZE

    train_dataset = YOLODataset(
        config.DATASET + "/train.csv",
        transform=None,  # config.train_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    _, target = train_dataset[3]
    target = target[0].unsqueeze(0)  # target[0] if S=13
    target = torch.cat([target, target] * 10)

    # anchor
    anchor = S * train_dataset.anchors[0]

    print(yl(predictions, target, anchor))
