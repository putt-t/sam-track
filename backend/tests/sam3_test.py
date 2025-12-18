import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


device = "cuda" if torch.cuda.is_available() else "cpu"


image_path = "lab_test.png"
prompt = "labrador"

model = build_sam3_image_model().to(device)
model.eval()

processor = Sam3Processor(model)

image = Image.open(image_path).convert("RGB")
inference_state = processor.set_image(image)

output = processor.set_text_prompt(state=inference_state, prompt=prompt)

masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

print("device:", device)
print("image:", image_path)
print("prompt:", prompt)
print("masks:", getattr(masks, "shape", None), getattr(masks, "dtype", None))
print("boxes:", getattr(boxes, "shape", None), getattr(boxes, "dtype", None))
print("scores:", getattr(scores, "shape", None), getattr(scores, "dtype", None))
