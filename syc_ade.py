#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 20:51:05 2025

@author: lenovo
"""

from transformers import AutoImageProcessor, UperNetForSemanticSegmentation, Mask2FormerForUniversalSegmentation
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import requests
import os 
from collections import namedtuple
import logging
from transformers import pipeline
logging.basicConfig(filename="logfile_icade.log", level = logging.INFO)


palette = np.asarray([
    [0, 0, 0],
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
])

 
model_id = "nlpconnect/vit-gpt2-image-captioning"
pipe_i2t = pipeline("image-to-text", model=model_id )

controlnet_model_path = "lllyasviel/sd-controlnet-seg"
controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
pipe.to("cuda")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

 
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()
#processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
#model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")

#image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
#model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-ade-semantic")


#image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
#model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")


#directory = "/home/lenovo/wzy/contseg/cityscapes/try" 
directory = "/home/lenovo/wzy/contseg/ADEChallengeData2016/images/training"
directory_ss = "/home/lenovo/wzy/contseg/ADEChallengeData2016/images/training_fenzu"  
#directory = "/home/lenovo/wzy/contseg/ade/images/training"  
k=0
for root, dirs, files in os.walk(directory_ss):
        for file in files:    
                save_gen = os.path.join(directory, f"{file}")
                
 
                save_gen = save_gen.replace(".jpg", "_syc.jpg")
                save_gen = save_gen.replace("/training", "/training_syc2") 
                               
 
                if os.path.exists(save_gen): 
                     k=k+1               
                     continue
 
                print(k)     
                image_path = os.path.join(root, file)                     
 
                image_path_gt = image_path.replace(".jpg", ".png") 
                
                subfoldername= os.path.basename(root)
                ls = os.path.join("images/training_fenzu", f"{subfoldername}")
 
 
                image_path_gt = image_path_gt.replace(ls, "annotations/training")
 
 
                img_gt = Image.open(image_path_gt).convert("L")
 
                W, H = img_gt.size
 
                W_n = W - (W % 8) if W % 8 == 0 else W - (W % 8)
                H_n = H - (H % 8) if H % 8 == 0 else H - (H % 8)
                
                left  = (W - W_n) / 2
                upper = (H - H_n) / 2
                right = (W + W_n) / 2
                lower = (H + H_n) / 2 
                
                cropped_img_gt = img_gt.crop((left, upper, right, lower))
 
                image_path_gt = save_gen.replace("_syc.jpg", "_syc.png") 
                cropped_image_path_gt = image_path_gt.replace("/images/training_syc2", "/annotations/training_syc")
               
                cropped_img_gt.save(cropped_image_path_gt)
 
 
                img_gt_nu = np.array(cropped_img_gt)
        
 
              
 
                color_gt = np.zeros((img_gt_nu.shape[0], img_gt_nu.shape[1], 3), dtype=np.uint8) # height, width, 3

                for label, color in enumerate(palette):
                    color_gt[img_gt_nu == label, :] = color
                #print('color_gt',color_gt)
                color_gt = color_gt.astype(np.uint8)
                
                image_color_gt = Image.fromarray(color_gt)
        
                #print(image_path)
                ip_adapter_image = Image.open(image_path).convert('RGB')
                ip_adapter_image = ip_adapter_image.crop((left, upper, right, lower))
                
                
 
                
                outputs_ic = pipe_i2t(ip_adapter_image)
                
                prompt = outputs_ic[0]["generated_text"]
 
                logging.info(f"prompt:{prompt}")
 
                image = pipe(
                           prompt=prompt,
                           image=image_color_gt,
                           ip_adapter_image=ip_adapter_image,
                           negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                           num_inference_steps=40,
                          ).images[0]
 
 

                #print(save_gen)
                
                logging.info(f"save_gen:{save_gen}")
                image.save(save_gen)
                #print(sss)
                
                #print(save_gen)
 
 
