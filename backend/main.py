import io
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

app = FastAPI(title="SAM3 Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"loading model on {device}...")
model = build_sam3_image_model().to(device)
model.eval()
processor = Sam3Processor(model)
print("model loaded")


def create_visualization(image: Image.Image, masks: torch.Tensor, boxes: torch.Tensor) -> Image.Image:
    img_array = np.array(image)
    overlay = img_array.copy()
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(masks), 3), dtype=np.uint8)

    for i, mask in enumerate(masks):
        mask_np = mask[0].cpu().numpy()

        mask_resized = Image.fromarray(mask_np).resize(image.size, Image.NEAREST)
        mask_resized = np.array(mask_resized)
        color_mask = np.zeros_like(img_array)
        color_mask[mask_resized > 0] = colors[i]

        overlay = np.where(
            mask_resized[..., np.newaxis] > 0,
            (overlay * 0.6 + color_mask * 0.4).astype(np.uint8),
            overlay
        )

    result = Image.fromarray(overlay)

    draw = ImageDraw.Draw(result)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.cpu().numpy()
        scale_x = image.width / 2048
        scale_y = image.height / 2048
        x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y

        draw.rectangle([x1, y1, x2, y2], outline=tuple(colors[i].tolist()), width=3)

    return result


@app.get("/")
async def root():
    return {"message": "api running", "device": device}


@app.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    prompt: str = Form(default="object")
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    num_detections = masks.shape[0] if len(masks.shape) > 0 else 0

    if num_detections == 0:
        result_image = image
    else:
        result_image = create_visualization(image, masks, boxes)

    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(
        img_byte_arr,
        media_type="image/png",
        headers={
            "X-Detections": str(num_detections),
            "X-Prompt": prompt
        }
    )


@app.post("/segment-comparison")
async def segment_comparison(
    file: UploadFile = File(...),
    prompt: str = Form(default="object")
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]

    num_detections = masks.shape[0] if len(masks.shape) > 0 else 0

    if num_detections == 0:
        segmented_image = image.copy()
    else:
        segmented_image = create_visualization(image, masks, boxes)

    total_width = image.width * 2
    comparison = Image.new('RGB', (total_width, image.height))
    comparison.paste(image, (0, 0))
    comparison.paste(segmented_image, (image.width, 0))

    draw = ImageDraw.Draw(comparison)
    draw.text((10, 10), "Original", fill=(255, 255, 255))
    draw.text((image.width + 10, 10), f"Segmented ({num_detections} detected)", fill=(255, 255, 255))

    img_byte_arr = io.BytesIO()
    comparison.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(
        img_byte_arr,
        media_type="image/png",
        headers={
            "X-Detections": str(num_detections),
            "X-Prompt": prompt
        }
    )
