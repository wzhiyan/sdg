from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
# from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ptp_utils
import seq_aligner
import cv2
import json
import torchvision
import argparse
import multiprocessing as mp
import torch.nn as nn
import threading
from random import choice
import random
import os
from distutils.version import LooseVersion
import argparse
from IPython.display import Image, display
from pytorch_lightning import seed_everything
from tqdm import tqdm
from dataset import *
#from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline
#from model.diffusers.models.unet_2d_condition import UNet2DConditionModel
#from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
#from model.unet import UNet2D,get_feature_dic,clear_feature_dic
from model.segment.transformer_decoder import seg_decorder
from model.decoder_ll  import seg_decorder_ll
import torch.optim as optim
import torch.nn.functional as F
from model.segment.criterion import SetCriterion
from model.segment.matcher import HungarianMatcher
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
import yaml
from utils import mask_image
from torch.optim.lr_scheduler import StepLR
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
import sd3_impls

from other_impls import SD3Tokenizer, SDClipModel, SDXLClipG, T5XXLModel
from PIL import Image
from safetensors import safe_open
from sd3_impls import (
    SDVAE,
    BaseModel,
    CFGDenoiser,
    SD3LatentFormat,
    SkipLayerCFGDenoiser,
)
from tqdm import tqdm
import math


LOW_RESOURCE = False 


def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in ckpt.keys():
        model_key = key
        if remap is not None and key in remap:
            model_key = remap[key]
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix) :].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(
                            f"Skipping key '{model_key}' in safetensors file as '{p}' does not exist in python model"
                        )
                        break
            if obj is None:
                continue
            try:
                tensor = ckpt.get_tensor(key).to(device=device)
                if dtype is not None and tensor.dtype != torch.int32:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                # print(f"K: {model_key}, O: {obj.shape} T: {tensor.shape}")
                if obj.shape != tensor.shape:
                    print(
                        f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}"
                    )
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e


CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}


class ClipG:
    def __init__(self, model_folder: str, device: str = "cpu"):
        with safe_open(
            f"{model_folder}/clip_g.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = SDXLClipG(CLIPG_CONFIG, device=device, dtype=torch.float32)
            load_into(f, self.model.transformer, "", device, torch.float32)


CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}


class ClipL:
    def __init__(self, model_folder: str):
        with safe_open(
            f"{model_folder}/clip_l.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device="cpu",
                dtype=torch.float32,
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            load_into(f, self.model.transformer, "", "cpu", torch.float32)


T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}


class T5XXL:
    def __init__(self, model_folder: str, device: str = "cpu", dtype=torch.float32):
        with safe_open(
            f"{model_folder}/t5xxl.safetensors", framework="pt", device="cpu"
        ) as f:
            self.model = T5XXLModel(T5_CONFIG, device=device, dtype=dtype)
            load_into(f, self.model.transformer, "", device, dtype)


class SD3:
    def __init__(
        self, model, shift, control_model_file=None, verbose=False, device="cpu"
    ):

        # NOTE 8B ControlNets were trained with a slightly different forward pass and conditioning,
        # so this is a flag to enable that logic.
        self.using_8b_controlnet = False

        with safe_open(model, framework="pt", device="cpu") as f:
            control_model_ckpt = None
            if control_model_file is not None:
                control_model_ckpt = safe_open(
                    control_model_file, framework="pt", device=device
                )
            self.model = BaseModel(
                shift=shift,
                file=f,
                prefix="model.diffusion_model.",
                device="cuda",
                dtype=torch.float16,
                control_model_ckpt=control_model_ckpt,
                verbose=verbose,
            ).eval()
 
            #self.model = DataParallel(self.model).cuda() 

            load_into(f, self.model, "model.", "cuda", torch.float16)
        if control_model_file is not None:
            control_model_ckpt = safe_open(
                control_model_file, framework="pt", device=device
            )
            self.model.control_model = self.model.control_model.to(device)
            load_into(
                control_model_ckpt,
                self.model.control_model,
                "",
                device,
                dtype=torch.float16,
                remap=CONTROLNET_MAP,
            )

            self.using_8b_controlnet = (
                self.model.control_model.y_embedder.mlp[0].in_features == 2048
            )
            self.model.control_model.using_8b_controlnet = self.using_8b_controlnet
        control_model_ckpt = None


class VAE:
    def __init__(self, model, dtype: torch.dtype = torch.float16):
        with safe_open(model, framework="pt", device="cpu") as f:
            self.model = SDVAE(device="cpu", dtype=dtype).eval().cpu()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, "cpu", dtype)


SHIFT = 3.0
# Naturally, adjust to the width/height of the model you have
WIDTH = 1024
HEIGHT = 1024
# Pick your prompt
PROMPT = "a photo of a cat"
# Most models prefer the range of 4-5, but still work well around 7
CFG_SCALE = 4.5
# Different models want different step counts but most will be good at 50, albeit that's slow to run
# sd3_medium is quite decent at 28 steps
STEPS = 50
# Seed
SEED = 23
# SEEDTYPE = "fixed"
SEEDTYPE = "rand"
# SEEDTYPE = "roll"
# Actual model file path
# MODEL = "models/sd3_medium.safetensors"
# MODEL = "models/sd3.5_large_turbo.safetensors"
MODEL = "models/sd3.5_medium.safetensors"  #"models/sd3.5_large.safetensors"
# VAE model file path, or set None to use the same model file
VAEFile = None  # "models/sd3_vae.safetensors"
# Optional init image file path
INIT_IMAGE = None
# ControlNet
CONTROLNET_COND_IMAGE = None
# If init_image is given, this is the percentage of denoising steps to run (1.0 = full denoise, 0.0 = no denoise at all)
DENOISE = 0.8
# Output file path
OUTDIR = "outputs"
# SAMPLER
SAMPLER = "dpmpp_2m"
# MODEL FOLDER
MODEL_FOLDER = "models"

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            if self.activate:
                self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        

class dict2obj(object):
    def __init__(self, d):
        self.__dict__['d'] = d
 
    def __getattr__(self, key):
        value = self.__dict__['d'][key]
        if type(value) == type({}):
            return dict2obj(value)
 
        return value

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if self.activate:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
    #         if attn.shape[1] <= 128 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if self.activate:
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]
            self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.activate = True


class SD3Inferencer:

    def __init__(self):
        self.verbose = False

    def print(self, txt):
        if self.verbose:
            print(txt)

    def load(
        self,
        model= "models/sd3.5_medium.safetensors",
        vae=None,  #None VAEFile
        shift= 3.0, #SHIFT,
        controlnet_ckpt=None,
        model_folder: str =  "models", #MODEL_FOLDER,
        text_encoder_device: str = "cpu",
        verbose=False,
        load_tokenizers: bool = True,
    ):
        self.verbose = verbose
        print("Loading tokenizers...")
        # NOTE: if you need a reference impl for a high performance CLIP tokenizer instead of just using the HF transformers one,
        # check https://github.com/Stability-AI/StableSwarmUI/blob/master/src/Utils/CliplikeTokenizer.cs
        # (T5 tokenizer is different though)
        self.tokenizer = SD3Tokenizer()
        if load_tokenizers:
            print("Loading Google T5-v1-XXL...")
            self.t5xxl = T5XXL(model_folder, text_encoder_device, torch.float32)
            print("Loading OpenAI CLIP L...")
            self.clip_l = ClipL(model_folder)
            print("Loading OpenCLIP bigG...")
            self.clip_g = ClipG(model_folder, text_encoder_device)
        print(f"Loading SD3 model {os.path.basename(model)}...")
        self.sd3 = SD3(model, shift, controlnet_ckpt, verbose, "cuda")
        print("Loading VAE model...")
        self.vae = VAE(vae or model)
        print("Models loaded.")

    def get_empty_latent(self, batch_size, width, height, seed, device="cuda"):
        self.print("Prep an empty latent...")
        shape = (batch_size, 16, height // 8, width // 8)
        latents = torch.zeros(shape, device=device)
        for i in range(shape[0]):
            prng = torch.Generator(device=device).manual_seed(int(seed + i))
            latents[i] = torch.randn(shape[1:], generator=prng, device=device)
        return latents

    def get_sigmas(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)

    def get_noise(self, seed, latent):
        generator = torch.manual_seed(seed)
        self.print(
            f"dtype = {latent.dtype}, layout = {latent.layout}, device = {latent.device}"
        )
        return torch.randn(
            latent.size(),
            dtype=torch.float32,
            layout=latent.layout,
            generator=generator,
            device="cpu",
        ).to(latent.dtype)

    def get_cond(self, prompt):
        self.print("Encode prompt...")
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
        g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
        t5_out, t5_pooled = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        return torch.cat([lg_out, t5_out], dim=-2), torch.cat(
            (l_pooled, g_pooled), dim=-1
        )

    def max_denoise(self, sigmas):
        max_sigma = float(self.sd3.model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def fix_cond(self, cond):
        cond, pooled = (cond[0].half().cuda(), cond[1].half().cuda())
        return {"c_crossattn": cond, "y": pooled}

    def do_sampling(
        self,
        latent,
        seed,
        conditioning,
        neg_cond,
        steps,
        cfg_scale,
        sampler="dpmpp_2m",
        controlnet_cond=None,
        denoise=1.0,
        skip_layer_config={},
    ) -> torch.Tensor:
        self.print("Sampling...")
        latent = latent.half().cuda()
        self.sd3.model = self.sd3.model.cuda()
        noise = self.get_noise(seed, latent).cuda()
        sigmas = self.get_sigmas(self.sd3.model.model_sampling, steps).cuda()
        sigmas = sigmas[int(steps * (1 - denoise)) :]
        conditioning = self.fix_cond(conditioning)
        neg_cond = self.fix_cond(neg_cond)
        extra_args = {
            "cond": conditioning,
            "uncond": neg_cond,
            "cond_scale": cfg_scale,
            "controlnet_cond": controlnet_cond,
        }
        print('noise',noise.shape)
        noise_scaled = self.sd3.model.model_sampling.noise_scaling(
            sigmas[0], noise, latent, self.max_denoise(sigmas)
        )
        print('noise_scaled',noise_scaled.shape)
        print('sigmas',sigmas.shape)
      
        sample_fn = getattr(sd3_impls, f"sample_{sampler}")
        #print('sample_fn',sample_fn) #<function sample_dpmpp_2m at 0x7fed8f9185e0>
        denoiser = (
            SkipLayerCFGDenoiser
            if skip_layer_config.get("scale", 0) > 0
            else CFGDenoiser
        )
        print('denoiser',denoiser)
        #print('self.sd3.model',self.sd3.model)
        ddd=denoiser(self.sd3.model, steps, skip_layer_config)
        print('ddd',ddd)
        latent,outputatt_sss,denoised_ss = sample_fn(
            denoiser(self.sd3.model, steps, skip_layer_config),
            noise_scaled,
            sigmas,
            extra_args=extra_args,
        )
 
        latent = SD3LatentFormat().process_out(latent)
        print('latent22',latent.shape)
        self.sd3.model = self.sd3.model.cpu()
        self.print("Sampling done")
        return latent

    def vae_encode(
        self, image, using_2b_controlnet: bool = False, controlnet_type: int = 0
    ) -> torch.Tensor:
        self.print("Encoding image to latent...")
        #image = image.convert("RGB")
        #image_np = np.array(image).astype(np.float32) / 255.0
        #image_np = np.moveaxis(image, 2, 0)
        #print('image_np',image_np.shape)
        #batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
        #image_torch = torch.from_numpy(batch_images).cuda()
        image_torch = image
        ####################print('image_torch',image_torch)
        if using_2b_controlnet:
            image_torch = image_torch * 2.0 - 1.0
        elif controlnet_type == 1:  # canny
            image_torch = image_torch * 255 * 0.5 + 0.5
        else:
            image_torch = 2.0 * image_torch - 1.0
        image_torch = image_torch.cuda()
        self.vae.model = self.vae.model.cuda()
        latent = self.vae.model.encode(image_torch).cpu()
        self.vae.model = self.vae.model.cpu()
        self.print("Encoded")
        return latent

    def vae_encode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.unsqueeze(0)
        latent = SD3LatentFormat().process_in(latent)
        return latent

    def vae_decode(self, latent) -> Image.Image:
        self.print("Decoding latent to image...")
        latent = latent.cuda()
        self.vae.model = self.vae.model.cuda()
        image = self.vae.model.decode(latent)
        image = image.float()
        self.vae.model = self.vae.model.cpu()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
        decoded_np = decoded_np.astype(np.uint8)
        out_image = Image.fromarray(decoded_np)
        self.print("Decoded")
        return out_image

    def _image_to_latent(
        self,
        image,
        width,
        height,
        using_2b_controlnet: bool = False,
        controlnet_type: int = 0,
    ) -> torch.Tensor:
        #image_data = Image.open(image)
        #image_data = image_data.resize((width, height), Image.LANCZOS)
        latent = self.vae_encode(image, using_2b_controlnet, controlnet_type)
        latent = SD3LatentFormat().process_in(latent)
        return latent

    def gen_image(
        self,
        prompts=None,# [PROMPT],
        width=WIDTH,
        height=HEIGHT,
        steps=STEPS,
        cfg_scale=CFG_SCALE,
        sampler=SAMPLER,
        seed=SEED,
        seed_type=SEEDTYPE,
        out_dir=OUTDIR,
        controlnet_cond_image=CONTROLNET_COND_IMAGE,
        init_image=INIT_IMAGE,
        denoise=DENOISE,
        skip_layer_config={},
    ):
        controlnet_cond = None
        if init_image:
            latent = self._image_to_latent(init_image, width, height)
        else:
            latent = self.get_empty_latent(1, width, height, seed, "cpu")
            latent = latent.cuda()
        print('latent',latent.shape)   
        if controlnet_cond_image:
            using_2b, control_type = False, 0
            if self.sd3.model.control_model is not None:
                using_2b = not self.sd3.using_8b_controlnet
                control_type = int(self.sd3.model.control_model.control_type.item())
            controlnet_cond = self._image_to_latent(
                controlnet_cond_image, width, height, using_2b, control_type
            )
        neg_cond = self.get_cond("")
        seed_num = None
        pbar = tqdm(enumerate(prompts), total=len(prompts), position=0, leave=True)
        #print('pbar',pbar)
        for i, prompt in pbar:
            if seed_type == "roll":
                seed_num = seed if seed_num is None else seed_num + 1
            elif seed_type == "rand":
                seed_num = torch.randint(0, 100000, (1,)).item()
            else:  # fixed
                seed_num = seed
            conditioning = self.get_cond(prompt)
            print('conditioning',conditioning[0].shape)
            print('conditioning',conditioning[1].shape)
            print('neg_cond',neg_cond[0].shape)
            print('neg_cond',neg_cond[1].shape)
            #print('sampler',sampler)
            sampled_latent = self.do_sampling(
                latent,
                seed_num,
                conditioning,
                neg_cond,
                steps,
                cfg_scale,
                sampler,
                controlnet_cond,
                denoise if init_image else 1.0,
                skip_layer_config,
            )
            print('sampled_latent',sampled_latent.shape)
            
            img_out = sampled_latent[0,:,:,:].clone()
            
            img_out = img_out.mean(axis=0).cpu().numpy()
            #img_out=F.interpolate(img_out, (h, w), mode="bicubic")
            cv2.normalize(img_out, img_out, 0, 255, cv2.NORM_MINMAX)
            img_out = np.array(img_out, dtype=np.uint8)
            img_out = cv2.applyColorMap(img_out,cv2.COLORMAP_JET)
            #print(img_out.shape)
            #save_namecc = f"{args.work_dir}/cc/{base_filename}.jpg"       
            #cv2.imwrite('sasa.jpg',img_out)      
            
            
            
            image = self.vae_decode(sampled_latent)
 
            save_path = os.path.join(out_dir, f"{i:06d}.png")
            self.print(f"Saving to to {save_path}")
            image.save(save_path)
            self.print("Done")




        
def freeze_params(params):
    for param in params:
        param.requires_grad = False
        
def semantic_inference(mask_cls, mask_pred):
    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
#         print(mask_cls.shape)
#         mask_cls = F.softmax(mask_cls, dim=-1)
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

    for i in range(1,semseg.shape[0]):
        if (semseg[i]*(semseg[i]>0.5)).sum()<5000:
            semseg[i] = 0

    return semseg

def instance_inference(mask_cls, mask_pred,class_n = 2,test_topk_per_image=20,query_n = 100):
    # mask_pred is already processed to have the same shape as original input
    image_size = mask_pred.shape[-2:]

    # [Q, K]
    scores = F.softmax(mask_cls, dim=-1)[:, :-1]
    labels = torch.arange(class_n , device=mask_cls.device).unsqueeze(0).repeat(query_n, 1).flatten(0, 1)
    # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
    scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
    labels_per_image = labels[topk_indices]

    topk_indices = topk_indices // class_n
    # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
#     print(topk_indices)
    mask_pred = mask_pred[topk_indices]


    result = Instances(image_size)
    # mask (before sigmoid)
    result.pred_masks = (mask_pred > 0).float()
    result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
    # Uncomment the following to get boxes from masks (this is slow)
    # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

    # calculate average mask prob
    mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
    result.scores = scores_per_image * mask_scores_per_image
    result.pred_classes = labels_per_image
    return result
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        nargs="?",
        default="./config/",
        help="config for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--image_limitation",
        type=int,
        default=5,
        help="image_limitation",
    )
    parser.add_argument(
        "--dataset", type=str, default="Cityscapes", help="dataset: VOC/Cityscapes/MaskCut"
    )
    parser.add_argument(
        "--save_name",
        type=str,
        help="the save dir name",
        default="Test"
    )
    opt = parser.parse_args()
    seed_everything(opt.seed)
    
    f = open(opt.config)
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = dict2obj(cfg)
    
    opt.dataset = cfg.DATASETS.dataset
    opt.batch_size = cfg.DATASETS.batch_size
#     opt.image_limitation = cfg.DATASETS.image_limitation
    
    # dataset
    if opt.dataset == "VOC":
        dataset = Semantic_VOC(
            set="train",
        )
    elif opt.dataset == "Cityscapes":
        dataset = Semantic_Cityscapes(
            set="train",image_limitation = opt.image_limitation
        )
    elif opt.dataset == "DeepFashion":
        dataset = Semantic_DeepFashion(
            set="train",image_limitation = opt.image_limitation
        )
    elif opt.dataset == "COCO":
        dataset = Instance_COCO(
            set="train",image_limitation = opt.image_limitation
        )   
    else:
        return
    
    print('dataset',dataset)
 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
 
    
    print('***********************   begin   **********************************')
    save_dir = 'checkpoint'
    os.makedirs(save_dir, exist_ok=True)
    learning_rate = cfg.SOLVER.learning_rate
    adam_weight_decay = cfg.SOLVER.adam_weight_decay
    total_epoch = cfg.SOLVER.total_epoch
    
    ckpt_dir = os.path.join(save_dir, opt.save_name)
    os.makedirs(ckpt_dir, exist_ok=True)
 
    
    inferencer = SD3Inferencer()
    model=MODEL 
    vae=VAEFile
    _shift=3
    controlnet_ckpt=None
    model_folder=MODEL_FOLDER
    text_encoder_device="cpu"
    verbose=False
    
    with torch.no_grad():
         inferencer.load(
             model,
             vae,
             _shift,
             controlnet_ckpt,
             model_folder,
             text_encoder_device,
             verbose,
             )  
         sd3 =SD3(model, _shift, controlnet_ckpt, verbose, "cuda")
 
    print('sd3',sd3)
    matcher = HungarianMatcher(
        cost_class=cfg.SEG_Decorder.CLASS_WEIGHT,
        cost_mask=cfg.SEG_Decorder.MASK_WEIGHT,
        cost_dice=cfg.SEG_Decorder.DICE_WEIGHT,
        num_points=cfg.SEG_Decorder.TRAIN_NUM_POINTS,
    )
        
    losses = ["labels", "masks"]
    weight_dict = {"loss_ce": cfg.SEG_Decorder.CLASS_WEIGHT, "loss_mask": cfg.SEG_Decorder.MASK_WEIGHT, "loss_dice": cfg.SEG_Decorder.DICE_WEIGHT}
    criterion = SetCriterion(
            cfg.SEG_Decorder.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=cfg.SEG_Decorder.no_object_weight,
            losses=losses,
            num_points=cfg.SEG_Decorder.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.SEG_Decorder.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.SEG_Decorder.IMPORTANCE_SAMPLE_RATIO,
        )
    
    
    seg_model=seg_decorder(num_classes=cfg.SEG_Decorder.num_classes, 
                           num_queries=cfg.SEG_Decorder.num_queries).to(device)
    #noise_scheduler = DDPMScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    
    os.makedirs(os.path.join(ckpt_dir, 'training'), exist_ok=True)
    print("learning_rate:",learning_rate)
    g_optim = optim.Adam(
            [{"params": seg_model.parameters()},],
            lr=learning_rate
          )
    scheduler = StepLR(g_optim, step_size=350, gamma=0.1)
    
    
    start_code = None
    
    LOW_RESOURCE = cfg.Diffusion.LOW_RESOURCE
    NUM_DIFFUSION_STEPS = cfg.Diffusion.NUM_DIFFUSION_STEPS
    GUIDANCE_SCALE = cfg.Diffusion.GUIDANCE_SCALE
    MAX_NUM_WORDS = cfg.Diffusion.MAX_NUM_WORDS
    
    steps=NUM_DIFFUSION_STEPS
    CFG_SCALE = 4.5
    sampler="dpmpp_2m"    
    controlnet_cond = None
    denoise=1.0
    skip_layer_config={}
    #controller = AttentionStore()
    #ptp_utils.register_attention_control(unet, controller)
    
    for j in range(total_epoch):
        print('Epoch ' +  str(j) + '/' + str(total_epoch))
        
        for step, batch in enumerate(dataloader):
            g_cpu = torch.Generator().manual_seed(random.randint(1, 10000000))
            
            # clear all features and attention maps
            #clear_feature_dic()
            #controller.reset()
            image = batch["image"]
            instances = batch["instances"]
            prompts = batch["prompt"]
            class_name = batch["classes_str"]
            original_image = batch["original_image"]
            sem_mask = batch["sem_mask"]
            # [1, 19, 512, 512]
#             print(instances["gt_masks"].shape)

            batch_size = image.shape[0]
            #print('image',image.shape)
            latent = inferencer._image_to_latent(image, WIDTH, HEIGHT)
            seed_num = torch.randint(0, 100000, (1,)).item()
            #print('prompts',prompts)
            prom=prompts[0]
            #print('prom',prom)
 

       
            conditioning = inferencer.get_cond(prom)
            neg_cond = inferencer.get_cond("")
            
            latent = latent.half().cuda()
            noise = inferencer.get_noise(seed_num, latent).cuda()
            
            sigmas = inferencer.get_sigmas(sd3.model.model_sampling, steps).cuda()
            sigmas = sigmas[int(steps * (1 - denoise)) :]
            conditioning = inferencer.fix_cond(conditioning)
            neg_cond = inferencer.fix_cond(neg_cond)
            extra_args = {
              "cond": conditioning,
              "uncond": neg_cond,
              "cond_scale": CFG_SCALE,
              "controlnet_cond": controlnet_cond,
            }          
            noise_scaled = sd3.model.model_sampling.noise_scaling(
               sigmas[-1], noise, latent, inferencer.max_denoise(sigmas)
            )
            sample_fn = getattr(sd3_impls, f"sample_{sampler}")
            
            denoiser = (
                SkipLayerCFGDenoiser
                if skip_layer_config.get("scale", 0) > 0
                else CFGDenoiser
            )           
            latent,outputatt_sss,denoised_ss = sample_fn(
                  denoiser(sd3.model, steps, skip_layer_config),
                  noise_scaled,
                  sigmas,
                  extra_args=extra_args,
                  )      
            latent = SD3LatentFormat().process_out(latent)
            #sd3.model = sd3.model.cpu()
            
            images_here = inferencer.vae_decode(latent)
            # save_path =f"img00g/img_{step+1}.jpg"
            # images_here.save(save_path)
            #print('outputatt_sss',outputatt_sss.shape)
            diffusion_features_ =outputatt_sss ##denoised_ss##[-1]
            diffusion_features = diffusion_features_
            diffusion_features_cat = torch.cat(diffusion_features,dim=1)
            #print('diffusion_features',diffusion_features.shape)

            outputs=seg_model(diffusion_features_cat)
            #break
            #print('outputs',outputs)
            #print('outputs',outputs.shape)  #([1, 100, 20])
 
            
            # bipartite matching-based loss
    
            losses = criterion(outputs, batch)
            loss = losses['loss_ce']*cfg.SEG_Decorder.CLASS_WEIGHT + losses['loss_mask']*cfg.SEG_Decorder.MASK_WEIGHT + losses['loss_dice']*cfg.SEG_Decorder.DICE_WEIGHT
#             loss = loss_fn(total_pred_seg, seg)
            
            g_optim.zero_grad()
            print("Training step: {0:05d}/{1:05d}, loss: {2:0.4f}, loss_ce: {3:0.4f},loss_mask: {4:0.4f},loss_dice: {5:0.4f}, lr: {6:0.6f}, prompt: ".format(step, len(dataloader), loss, losses['loss_ce']*cfg.SEG_Decorder.CLASS_WEIGHT,losses['loss_mask']*cfg.SEG_Decorder.MASK_WEIGHT, losses['loss_dice']*cfg.SEG_Decorder.DICE_WEIGHT,float(g_optim.state_dict()['param_groups'][0]['lr'])),prompts)
            loss.backward()
            g_optim.step()
            
            if step%100 ==0 or losses['loss_mask']*cfg.SEG_Decorder.MASK_WEIGHT<0.3:
                
                mask_cls_results = outputs["pred_logits"]
                mask_pred_results = outputs["pred_masks"]
                mask_pred_results = F.interpolate(
                                    mask_pred_results,
                                    size=(image.shape[-2], image.shape[-1]),
                                    mode="bilinear",
                                    align_corners=False,
                                    )
                
                for mask_cls_result, mask_pred_result in zip(mask_cls_results, mask_pred_results):

                    height = width = 512
                    if cfg.SEG_Decorder.task=="semantic":
                        label_pred_prob = retry_if_cuda_oom(semantic_inference)(mask_cls_result, mask_pred_result)
                        label_pred_prob = torch.argmax(label_pred_prob, axis=0)
                        label_pred_prob = label_pred_prob.cpu().numpy()
                        cv2.imwrite(os.path.join(ckpt_dir, 'training/'+'viz_sample_{0:05d}_seg'.format(step)+'.png'),label_pred_prob*100)
                        
                    elif cfg.SEG_Decorder.task=="instance":
                        instance_r = retry_if_cuda_oom(instance_inference)(mask_cls_result, mask_pred_result, cfg.SEG_Decorder.num_classes)
                        pred_masks = instance_r.pred_masks.cpu().numpy().astype(np.uint8)
                        pred_boxes = instance_r.pred_boxes
                        scores = instance_r.scores 
                        pred_classes = instance_r.pred_classes 
                        
                        vis_i  = original_image.cpu().numpy()[0].astype(np.uint8)
                        vis_i = np.array(vis_i,dtype=float)

                        for m in pred_masks:
#                             print(vis_i.shape)
                            vis_i,_ = mask_image(vis_i,m)
#                         print(pred_masks.shape)
                        cv2.imwrite(os.path.join(ckpt_dir, 'training/'+'viz_sample_{0:05d}_seg'.format(step)+'.jpg'),vis_i)
                        try:
                            gt_mask = instances["gt_masks"][0][0]
                            gt_mask = np.array(gt_mask)*255
                            cv2.imwrite(os.path.join(ckpt_dir, 'training/'+'viz_sample_{0:05d}_gt_seg'.format(step)+'.jpg'),gt_mask)
                        except:
                            pass
    
                
#             print(total_pred_seg.shape)
        print("Saving latest checkpoint to",ckpt_dir)
        torch.save(seg_model.state_dict(), os.path.join(ckpt_dir, 'latest_checkpoint.pth'))
        if j%10==0:
            torch.save(seg_model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_'+str(j)+'.pth'))
        scheduler.step()


if __name__ == "__main__":
    main()
