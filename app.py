import os

import gradio as gr

from src.run.yolov3.inference import YoloInfer


infer = YoloInfer(model_path="./checkpoint/model.pt")

demo = gr.Interface(
    fn=infer.infer,
    inputs=[
        gr.Image(
            shape=(416, 416),
            label="Input Image",
            value="./sample/bird_plane.jpeg",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            value=0.2,
            label="IOU Threshold",
            info="Permissible overlap for the same class bounding boxes",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            value=0.95,
            label="Objectness Threshold",
            info="Confidence for each pixel to predict an object",
        ),
        gr.Slider(
            minimum=0,
            maximum=1,
            value=0.5,
            label="Class Threshold",
            info="Confidence for each pixel to predict a class",
        ),
        gr.Slider(
            minimum=0,
            maximum=10,
            value=1,
            label="Font Size",
            info="Bounding box text size",
        ),
    ],
    outputs=[
        gr.Image(),
    ],
    examples=[
        [os.path.join("./sample/", f)]
        for f in os.listdir("./sample/")
    ],
)


demo.launch()
