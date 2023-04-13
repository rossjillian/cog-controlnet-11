from huggingface_hub import hf_hub_download

REPO = "lllyasviel/ControlNet-v1-1"
MODEL_PATHS = [
    "control_v11e_sd15_shuffle",
    "control_v11e_sd15_canny",
    "control_v11e_sd15_depth",
    "control_v11e_sd15_inpaint",
    "control_v11e_sd15_lineart",
    "control_v11e_sd15_mlsd",
    "control_v11e_sd15_normalbae",
    "control_v11e_sd15_openpose",
    "control_v11e_sd15_scribble",
    "control_v11e_sd15_seg",
    "control_v11e_sd15_softedge",
    "control_v11e_sd15_tile"
]

for model in MODEL_PATHS:
    hf_hub_download(repo_id=REPO, filename="%s.yaml" % model)
    hf_hub_download(repo_id=REPO, filename="%s.pth" % model)

hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5", filename="v1-5-pruned.ckpt")
