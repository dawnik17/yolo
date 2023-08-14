import os

import albumentations as A
import config
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

from src.loss.yolov3 import YoloLoss
from src.model.yolov3 import YOLOv3 as Model


class YoloInfer:
    def __init__(self, model_path):
        self.model = self.load_model(model_path=model_path)
        self.transform = A.Compose(
            [
                A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
                A.Normalize(
                    mean=[0, 0, 0],
                    std=[1, 1, 1],
                    max_pixel_value=255,
                ),
                ToTensorV2(),
            ]
        )

        self.scaled_anchors = (
            torch.tensor(config.ANCHORS) * torch.tensor(config.S).reshape(-1, 1, 1)
        ).to(config.DEVICE)

    def load_model(self, model_path):
        model = Model(num_classes=config.NUM_CLASSES).to(config.DEVICE)

        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=config.DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        return model

    @staticmethod
    def pred_to_boxes(prediction, anchors):
        """
        prediction tensor = [batch, num_anchors_per_scale, scale, scale, 5 + num_classes]
        5 = [objness, cx, cy, w, h]
        anchors tensor = [num_anchors_per_scale, 2]

        Note: The below operation could been done entirely inplace.
            Slightly unoptimsed implementation below to maintain readability

        Output shape: [batch, num_anchors_per_scale, scale, scale, 7]
        7: [predicted_class's_idx, obj score, cx, cy, width, height, predicted class probability score]
        """
        scale = prediction.shape[2]

        # reversing the equations of box loss and obj in the loss function
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        cx_cy = torch.sigmoid(prediction[..., 1:3])
        width_height = anchors * torch.exp(prediction[..., 3:5])

        # reversing the equations we wrote while making training data
        arange = torch.arange(scale, device=config.DEVICE)
        cx = (arange.reshape(1, 1, 1, scale, 1) + cx_cy[..., 0:1]) / scale
        cy = (arange.reshape(1, 1, scale, 1, 1) + cx_cy[..., 1:2]) / scale
        width_height = width_height / scale

        # class prediction
        class_predictions = torch.softmax(prediction[..., 5:], dim=-1)
        class_score, class_idx = torch.max(class_predictions, dim=-1)
        class_score, class_idx = class_score.unsqueeze(-1), class_idx.unsqueeze(-1)

        # objectness score
        obj_score = torch.sigmoid(prediction[..., 0:1])
        return torch.cat(
            [class_idx, obj_score, cx, cy, width_height, class_score], dim=-1
        )

    @staticmethod
    def sort_3Dtensor_rows_on_two_columns(
        tensor, index1, index2, descending1=True, descending2=True
    ):
        """
        tensor = tensor([[[1, 2, 3],
                        [1, 3, 4],
                        [0, 2, 1]],

                        [[0, 2, 3],
                        [1, 4, 5],
                        [0, 1, 2]]])

        sort_tensor_rows_on_two_columns(tensor,
                                        index1=0,
                                        index2=1,
                                        descending1=False,
                                        descending2=True)

        output = tensor([[[0, 2, 1],
                        [1, 3, 4],
                        [1, 2, 3]],

                        [[0, 2, 3],
                        [0, 1, 2],
                        [1, 4, 5]]])
        """
        inner_sorting = torch.argsort(tensor[..., index2], descending=descending1)
        inner_sorted = torch.gather(
            tensor, 1, inner_sorting.unsqueeze(-1).expand(-1, -1, tensor.size(2))
        )

        outer_sorting = torch.argsort(
            inner_sorted[:, :, index1], stable=True, descending=descending2
        )
        outer_sorted = torch.gather(
            inner_sorted,
            1,
            outer_sorting.unsqueeze(-1).expand(-1, -1, inner_sorted.size(2)),
        )
        return outer_sorted

    @staticmethod
    def non_max_supression(
        self, prediction, iou_threshold, object_threshold, class_threshold
    ):
        """
        prediction = [batch, summation(num_anchors_per_scale * scale * scale), 7]
        i.e. [batch, (3 * 13 * 13 + 3 * 26 * 26 + 3 * 52 * 52), 7]

        7: [class_pred, obj_score, cx, cy, width, height, class_score]
        """
        """
        inside each batch output,
        first sort by class prediction, 
        and inside each class sort objectness in descending 
        """
        prediction = self.sort_3Dtensor_rows_on_two_columns(
            tensor=prediction, index1=0, index2=1, descending1=True, descending2=True
        )

        """
        remove predictions with object threshold below the given threshold
        and split prediction to get a list of tensors
        
        length of list = batch size
        each element in the list = results/output of 1 image
        """
        # objectness condition [threshold]
        objectness = (prediction[..., 1] > object_threshold) & (
            prediction[..., 6] > class_threshold
        )
        indices = torch.nonzero(objectness)
        batch_boxes = torch.split(
            tensor=prediction[objectness],
            split_size_or_sections=torch.bincount(indices[:, 0]).tolist(),
            dim=0,
        )

        # iterate for output
        output = []

        for boxes in tqdm(batch_boxes, disable=False):
            # boxes shape = [-1, 7]
            boxes = boxes.tolist()
            final_boxes = []

            while boxes:
                top_box = boxes.pop(0)

                idx = 0

                while idx < len(boxes):
                    box = boxes[idx]

                    # class match
                    if box[0] != top_box[0]:
                        break

                    # iou match
                    if (
                        YoloLoss.iou(torch.tensor(top_box[2:6]), torch.tensor(box[2:6]))
                        > iou_threshold
                    ):
                        del boxes[idx]

                        idx -= 1

                    idx += 1

                final_boxes.append(top_box)

            output.append(final_boxes)

        return output

    @staticmethod
    def draw_bounding_boxes(image, boxes, font_size=1):
        """Draws bounding boxes on the image using OpenCV"""
        cmap = plt.get_cmap("tab20b")
        class_labels = (
            config.COCO_LABELS if config.DATASET == "COCO" else config.PASCAL_CLASSES
        )
        colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
        im = np.array(image)
        height, width, _ = im.shape

        # font = ImageFont.truetype("DejaVuSans.ttf", 20)  # Load the DejaVuSans font

        for box in boxes:
            assert (
                len(box) == 7
            ), "box should contain class pred, confidence, x, y, width, height, class score"
            class_pred = box[0]
            class_score = round(box[-1], 2)

            upper_left_x = int((box[2] - box[4] / 2) * width)
            upper_left_y = int((box[3] - box[5] / 2) * height)
            lower_right_x = int((box[2] + box[4] / 2) * width)
            lower_right_y = int((box[3] + box[5] / 2) * height)

            color = colors[int(class_pred)]
            color_rgb = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            thickness = max(
                int((0.0005 * (image.shape[0] + image.shape[1]) / 2) + 1), 1
            )

            cv2.rectangle(
                im,
                (upper_left_x, upper_left_y),
                (lower_right_x, lower_right_y),
                color_rgb,
                thickness=thickness,
            )

            # label
            font_scale = font_size
            label = f"{class_labels[int(class_pred)]} {class_score}"
            text_size = cv2.getTextSize(
                label,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                thickness=1,
            )[0]

            # Draw rectangle background
            cv2.rectangle(
                im,
                (upper_left_x, upper_left_y),
                (upper_left_x + text_size[0], upper_left_y - text_size[1]),
                color_rgb,
                thickness=-1,
            )
            cv2.putText(
                im,
                label,
                (upper_left_x, upper_left_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=[0, 0, 0],
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        return im

    def infer(
        self,
        image: np.array,
        iou_threshold=0.75,
        object_threshold=0.75,
        class_threshold=0.5,
    ):
        self.model.eval()
        input_tensor = self.transform(image=image)["image"].unsqueeze(0)

        with torch.no_grad():
            """
            output = list of tensors
            tensor shape=[batch, num_anchors_per_scale, scale, scale, 5 + num_classes]
            """
            output = self.model(input_tensor.to(config.DEVICE))

            # convert model prediction to actual box prediction
            output = torch.cat(
                [
                    self.pred_to_boxes(out, self.scaled_anchors[idx]).reshape(
                        out.shape[0], -1, 7
                    )
                    for idx, out in enumerate(output)
                ],
                dim=1,
            )

            # non max suppression
            output = self.non_max_supression(
                prediction=output,
                iou_threshold=iou_threshold,
                object_threshold=object_threshold,
                class_threshold=class_threshold,
            )

        return self.draw_bounding_boxes(image, output[0])

    @staticmethod
    def load_image_as_array(image_path):
        # Load a PIL image
        pil_image = Image.open(image_path)

        # Convert PIL image to NumPy array
        return np.array(pil_image.convert("RGB"))
    
    @staticmethod
    def plot_array(array: np.array, figsize=(10,10)):
        plt.figure(figsize=figsize)
        plt.imshow(array)
        plt.show()

    @staticmethod
    def save_numpy_as_image(numpy_array, image_path):
        """
        Saves a NumPy array as an image.
        Args:
            numpy_array (numpy.ndarray): The NumPy array to be saved as an image.
            image_path (str): The path where the image will be saved.
        """
        # Convert the NumPy array to a PIL image
        image = Image.fromarray(numpy_array)
        
        # Save the PIL image to the specified path
        image.save(image_path)