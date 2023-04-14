import torch
from typing import List
import numpy as np
from PIL import Image
import cv2
import time
import einops
import random
import os

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.midas import MidasDetector
from annotator.lineart import LineartDetector
from annotator.normalbae import NormalBaeDetector
from annotator.openpose import OpenposeDetector
from annotator.oneformer import OneformerADE20kDetector
from annotator.hed import HEDdetector

from cog import BasePredictor, Input, Path
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        st = time.time()
        # Canny
        canny_model_name = 'control_v11p_sd15_canny'
        self.canny_model = load_model(canny_model_name)
        self.canny_dim_sampler = DDIMSampler(self.canny_model)
        # Depth
        depth_model_name = 'control_v11p_sd15_depth'
        self.depth_model = load_model(depth_model_name)
        self.depth_dim_sampler = DDIMSampler(self.depth_model)
        # Normal
        normal_model_name = 'control_v11p_sd15_normalbae'
        self.normal_model = load_model(normal_model_name)
        self.normal_dim_sampler = DDIMSampler(self.normal_model)
        # Lineart
        lineart_model_name = 'control_v11p_sd15_lineart'
        self.lineart_model = load_model(lineart_model_name)
        self.lineart_dim_sampler = DDIMSampler(self.lineart_model)
        # Scribble
        scribble_model_name = 'control_v11p_sd15_scribble'
        self.scribble_model = load_model(scribble_model_name)
        self.scribble_dim_sampler = DDIMSampler(self.scribble_model)
        # Seg
        seg_model_name = 'control_v11p_sd15_seg'
        self.seg_model = load_model(seg_model_name)
        self.seg_dim_sampler = DDIMSampler(self.seg_model)
        # Pose
        pose_model_name = 'control_v11p_sd15_openpose'
        self.pose_model = load_model(pose_model_name)
        self.pose_dim_sampler = DDIMSampler(self.pose_model)
        print("Setup complete in %f" % (time.time() - st))

    @torch.inference_mode()
    def predict(self,
                image: Path = Input(
                    description="Input image"
                ),
                prompt: str = Input(
                    description="Prompt for the model"
                ),
                structure: str = Input(
                    description="Structure to condition on",
                    choices=["canny", "depth", "hed", "hough", "normal", "pose", "scribble", "seg"]
                ),
                num_samples: str = Input(
                    description="Number of samples (higher values may OOM)",
                    choices=['1', '4'],
                    default='1'
                ),
                image_resolution: str = Input(
                    description="Resolution of image (square)",
                    choices=['256', '512', '768'],
                    default='512'
                ),
                ddim_steps: int = Input(
                    description="Steps",
                    default=20
                ),
                strength: float = Input(
                    description="Control strength",
                    default=1.0
                ),
                scale: float = Input(
                    description="Scale for classifier-free guidance",
                    default=9.0,
                    ge=0.1,
                    le=30.0
                ),
                seed: int = Input(
                    description="Seed",
                    default=None
                ),
                eta: float = Input(
                    description="Controls the amount of noise that is added to the input data during the denoising diffusion process. Higher value -> more noise",
                    default=0.0
                ),
                preprocessor_resolution: int = Input(
                    description="Preprocessor resolution",
                    default=512
                ),
                a_prompt: str = Input(
                    description="Additional text to be appended to prompt",
                    default="Best quality, extremely detailed"
                ),
                n_prompt: str = Input(
                    description="Negative prompt",
                    default="Longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
                ),
                # Only applicable when model type is 'canny'
                low_threshold: int = Input(
                    description="[canny only] Line detection low threshold",
                    default=100,
                    ge=1,
                    le=255
                ),
                # Only applicable when model type is 'canny'
                high_threshold: int = Input(
                    description="[canny only] Line detection high threshold",
                    default=200,
                    ge=1,
                    le=255
                ),
                ) -> List[Path]:
        image = np.array(Image.open(image))
        image = HWC3(image)
        img = resize_image(image, image_resolution)
        H, W, C = img.shape

        model = self.select_model(structure)
        ddim_sampler = self.select_sampler(structure)
        detected_map = self.process_image(image, structure, preprocessor_resolution, low_threshold, high_threshold)

        detected_map = HWC3(detected_map)
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        cond = {"c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": [control],
                   "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        model.control_scales = [strength] * 13
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]

        output_paths = []
        for i, sample in enumerate(results):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))
        return output_paths

    def select_model(self, structure):
        if structure == 'canny':
            model = self.canny_model
        elif structure == 'depth':
            model = self.depth_model
        elif structure == 'lineart':
            model = self.lineart_model
        elif structure == 'normal':
            model = self.normal_model
        elif structure == 'pose':
            model = self.pose_model
        elif structure == 'scribble':
            model = self.scribble_model
        elif structure == 'seg':
            model = self.seg_model
        return model

    def process_image(self, image, structure, preprocessor_resolution, low_threshold=100, high_threshold=200):
        if structure == 'canny':
            input_image = self.canny_preprocessor(image, preprocessor_resolution, low_threshold, high_threshold)
        elif structure == 'depth':
            input_image = self.depth_preprocessor(image, preprocessor_resolution)
        elif structure == 'lineart':
            input_image = self.lineart_preprocessor(image, preprocessor_resolution)
        elif structure == 'normal':
            input_image = self.normal_preprocessor(image, preprocessor_resolution)
        elif structure == 'pose':
            input_image = self.pose_preprocessor(image, preprocessor_resolution)
        elif structure == 'scribble':
            input_image = self.scribble_preprocessor(image, preprocessor_resolution)
        elif structure == 'seg':
            input_image = self.seg_preprocessor(image, preprocessor_resolution)
        return input_image

    def canny_preprocessor(self, image, preprocessor_resolution, low_threshold, high_threshold):
        detected_map = CannyDetector(resize_image(image, preprocessor_resolution), low_threshold, high_threshold)
        return detected_map

    def depth_preprocessor(self, image, preprocessor_resolution):
        detected_map = MidasDetector(resize_image(image, preprocessor_resolution))
        return detected_map

    def lineart_preprocessor(self, image, preprocessor_resolution):
        detected_map = LineartDetector(resize_image(image, preprocessor_resolution), coarse='Coarse')
        return detected_map

    def normal_preprocessor(self, image, preprocessor_resolution):
        detected_map = NormalBaeDetector(resize_image(image, preprocessor_resolution))
        return detected_map

    def pose_preprocessor(self, image, preprocessor_resolution):
        detected_map = OpenposeDetector(resize_image(image, preprocessor_resolution), hand_and_face='Full')
        return detected_map

    def scribble_preprocessor(self, image, preprocessor_resolution):
        detected_map = HEDdetector(resize_image(image, preprocessor_resolution))
        return detected_map

    def seg_preprocessor(self, image, preprocessor_resolution):
        detected_map = OneformerADE20kDetector(resize_image(image, preprocessor_resolution))
        return detected_map


def load_model(name):
    model = create_model(f'./models/{name}.yaml').cpu()
    torch.load(os.path.abspath('./models/v1-5-pruned.ckpt'))
    model.load_state_dict(load_state_dict(f'./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
    model.load_state_dict(load_state_dict(f'./models/{name}.pth', location='cuda'), strict=False)
    model = model.cuda()
    return model
