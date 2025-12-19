import io
import os
import tempfile
import torch
import cv2
import numpy as np
import asyncio
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

progress_status = {}

app = FastAPI(title="SAM3 Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
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


@app.get("/progress/{session_id}")
async def get_progress(session_id: str):
    return progress_status.get(session_id, {"current": 0, "total": 0, "status": "not_found"})


@app.post("/segment-video")
async def segment_video(
    file: UploadFile = File(...),
    prompt: str = Form(default="object"),
    start_frame: int = Form(default=0),
    end_frame: int = Form(default=-1),
    session_id: str = Form(default="default")
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        content = await file.read()
        tmp_input.write(content)
        tmp_input_path = tmp_input.name

    cap = cv2.VideoCapture(tmp_input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if end_frame == -1 or end_frame > total_frames:
        end_frame = total_frames

    tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_output_path = tmp_output.name
    tmp_output.close()

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(tmp_output_path, fourcc, fps, (width, height))

    frame_idx = 0
    processed_count = 0
    total_to_process = end_frame - start_frame
    
    progress_status[session_id] = {"current": 0, "total": total_to_process, "status": "processing"}
    
    colors = None

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= start_frame:
            processed_count += 1
            progress_status[session_id] = {"current": processed_count, "total": total_to_process, "status": "processing"}
            await asyncio.sleep(0)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            inference_state = processor.set_image(pil_image)
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)

            masks = output["masks"]
            boxes = output["boxes"]
            num_detections = masks.shape[0] if len(masks.shape) > 0 else 0

            if num_detections > 0:
                if colors is None or len(colors) < num_detections:
                    np.random.seed(42)
                    colors = np.random.randint(0, 255, size=(num_detections, 3), dtype=np.uint8)
                
                overlay = frame_rgb.copy()

                for i, mask in enumerate(masks):
                    mask_np = mask[0].cpu().numpy()
                    mask_resized = Image.fromarray(mask_np).resize((width, height), Image.NEAREST)
                    mask_resized = np.array(mask_resized)
                    color_mask = np.zeros_like(overlay)
                    color_mask[mask_resized > 0] = colors[i]
                    overlay = np.where(
                        mask_resized[..., np.newaxis] > 0,
                        (overlay * 0.6 + color_mask * 0.4).astype(np.uint8),
                        overlay
                    )

                frame_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
        else:
            frame_bgr = frame

        out.write(frame_bgr)
        frame_idx += 1

    cap.release()
    out.release()
    os.unlink(tmp_input_path)
    
    progress_status[session_id] = {"current": processed_count, "total": total_to_process, "status": "complete"}

    return FileResponse(
        tmp_output_path,
        media_type="video/mp4",
        filename="segmented_video.mp4",
        headers={
            "X-Total-Frames": str(end_frame - start_frame),
            "X-Prompt": prompt
        }
    )


@app.post("/get-video-info")
async def get_video_info(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    os.unlink(tmp_path)

    return {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height
    }
