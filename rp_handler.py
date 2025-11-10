import os
import uuid
import logging
import gc
from datetime import timedelta
import torch
import runpod
from google.cloud import storage
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
import wan
from wan.utils.utils import save_video
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wan-t2v-serverless")

DEVICE = 0
RANK = 0
OFFLOAD_MODEL = True
PIPELINE = None
PIPELINE_CFG = WAN_CONFIGS["t2v-A14B"]
CKPT_DIR = "./Wan2.2-T2V-A14B"
LIGHTNING_DIR = "./Wan2.2-Lightning"
KEEP_LORA = "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0"
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

def fetch_gcs_json_from_drive(file_id: str) -> dict:
    """Download a JSON file from Google Drive given its file ID"""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def upload_to_gcs_public(source_file, bucket_name="runpod_bucket_testing"):
    # Fetch the GCS service account JSON from Drive
    gcs_json_dict = fetch_gcs_json_from_drive("1leNukepERYsBmoKSYTbqUjGb-pQvwQlz")  # Replace with your Drive file ID
    creds_path = "/tmp/gcs_creds.json"
    with open(creds_path, "w") as f:
        json.dump(gcs_json_dict, f)

    client = storage.Client.from_service_account_json(creds_path)
    bucket = client.bucket(bucket_name)
    destination_blob = f"t2v_videos/{uuid.uuid4()}.mp4"
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(source_file)
    url = blob.generate_signed_url(expiration=timedelta(hours=1))
    return url

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
            save_video_to_file(video, save_path, fps=PIPELINE_CFG.sample_fps)
            del video
            torch.cuda.synchronize()

        gcs_url = upload_to_gcs_public(save_path)
        return {"status": "success", "gcs_url": gcs_url, "seed": seed}

    except Exception as e:
        logger.exception("Generation failed")
        return {"status": "failed", "error": str(e)}

runpod.serverless.start({"handler": generate_t2v})
