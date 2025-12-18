import os
import uuid
import logging
import gc
import json
import torch
import runpod
import wan
from datetime import timedelta
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.utils import save_video
import requests
import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wan-t2v-serverless")

DEVICE = 0
RANK = 0
OFFLOAD_MODEL = True
PIPELINE = None
PIPELINE_CFG = WAN_CONFIGS["t2v-A14B"]
CKPT_DIR = "./Wan2.2-T2V-A14B"
LIGHTNING_DIR = "./Wan2.2-Lightning"
KEEP_LORA = "Wan2.2-T2V-A14B-4steps-lora-250928"
SAVE_DIR = "test_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# Download WAN checkpoint & Lightning repo
if not os.path.exists(LIGHTNING_DIR):
    os.system(f"huggingface-cli download lightx2v/Wan2.2-Lightning --local-dir {LIGHTNING_DIR}")
if not os.path.exists(CKPT_DIR):
    os.system(f"huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir {CKPT_DIR}")
for folder in os.listdir(LIGHTNING_DIR):
    folder_path = os.path.join(LIGHTNING_DIR, folder)
    if os.path.isdir(folder_path) and folder != KEEP_LORA:
        os.system(f"rm -rf {folder_path}")

# -------------------- AWS S3 Setup --------------------
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_KEY")
REGION = os.environ.get("AWS_REGION", "us-east-2")
BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME", "runpodstorageforserverless")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION
)

# -------------------- Model Setup --------------------
def get_pipeline():
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    PIPELINE = wan.WanT2V(
        config=PIPELINE_CFG,
        checkpoint_dir=CKPT_DIR,
        lora_dir=os.path.join(LIGHTNING_DIR, KEEP_LORA),
        device_id=DEVICE,
        rank=RANK,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        convert_model_dtype=True,
    )
    return PIPELINE

def save_video_to_file(video, save_path, fps):
    save_video(video[None], save_path, fps=fps, nrow=1, normalize=True, value_range=(-1, 1))

# -------------------- S3 Upload --------------------
def upload_to_s3_public(source_file, folder="t2v_videos"):
    """
    Upload a local file to S3 under the specified folder.
    Returns the public URL (bucket must allow public read via policy).
    """
    destination_key = f"{folder}/{uuid.uuid4()}.mp4"
    content_type = "video/mp4"

    s3.upload_file(
        Filename=source_file,
        Bucket=BUCKET_NAME,
        Key=destination_key,
        ExtraArgs={"ContentType": content_type}
    )

    public_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{destination_key}"
    return public_url

# -------------------- Generation Handler --------------------
def generate_t2v(job):
    try:
        inputs = job.get("input", {})
        prompt = inputs.get("prompt", "Two anthropomorphic cats in comfy boxing gear fight on a stage")
        size = inputs.get("size", "1280*720")
        frame_num = int(inputs.get("frame_num", 21))
        if size not in SUPPORTED_SIZES["t2v-A14B"]:
            return {"status":"error", "error": f"Unsupported size {size}"}

        pipeline = get_pipeline()
        seed = int(inputs.get("seed", torch.randint(0, 999999, (1,)).item()))

        with torch.no_grad():
            video = pipeline.generate(
                prompt,
                size=SIZE_CONFIGS[size],
                frame_num=frame_num,
                shift=5.0,
                sample_solver='euler',
                sampling_steps=4,
                guide_scale=(1.0,1.0),
                seed=seed,
                offload_model=OFFLOAD_MODEL
            )
            save_path = os.path.join(SAVE_DIR, f"t2v_{uuid.uuid4()}.mp4")
            save_video_to_file(video, save_path, fps=20)
            del video
            torch.cuda.synchronize()

        s3_url = upload_to_s3_public(save_path)
        return {"status": "success", "s3_url": s3_url, "seed": seed}

    except Exception as e:
        logger.exception("Generation failed")
        return {"status": "failed", "error": str(e)}

runpod.serverless.start({"handler": generate_t2v})
