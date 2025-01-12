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
from transformers import pipeline
import requests
import os 
from collections import namedtuple
import logging
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

Label = namedtuple( 'Label' , [
    'name'        ,  
    'id'          , 
    'trainId'     ,  
    'category'    ,  
    'categoryId'  ,  
    'hasInstances',  
    'ignoreInEval',  
    'color'       ,  
    ] )
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]


label2trainid   = { label.id : label.trainId for label in labels   }
trainId2name   = { label.trainId : label.name for label in labels   }


logging.basicConfig(filename="logfile_ip_i2t.log", level = logging.INFO)
 

model_id = "llava-hf/bakLlava-v1-hf"

pipe_i2t = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})
 

controlnet_model_path = "lllyasviel/sd-controlnet-seg"
controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
pipe.to("cuda")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

 
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
#directory = "/home/lenovo/wzy/contseg/cityscapes/try" 
directory = "/home/lenovo/wzy/contseg/cityscapes/gtFine_trainvaltest/gtFine/train" 
 
for root, dirs, files in os.walk(directory):
        for file in files:    
            if '_gtFine_color.png' in file:               
                image_path_label = os.path.join(root, file)
                image_path_gt = image_path_label.replace("gtFine_color.png", "gtFine_labelIds.png") 
                 
                img_label = Image.open(image_path_label).convert('RGB') 
                img_gt = Image.open(image_path_gt).convert("L")
                gtimage = np.array(img_gt)
                for k, v in label2trainid.items():
                    gtimage[gtimage == k] = v
                 
                #print('prompt',prompt)

                pipe_i2t = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

                 
                prompt = "USER: <image>\nDetaily imagine and describe the scene this image taken from? \nASSISTANT:"
               
 
                image_ip = image_path_label.replace("gtFine_trainvaltest/gtFine/train", "leftImg8bit_trainvaltest/leftImg8bit/train")
                image_ip = image_ip.replace("_gtFine_color.png", "_leftImg8bit.png")
                ip_adapter_image = Image.open(image_ip)
                
                
                outputs = pipe_i2t(ip_adapter_image, prompt=prompt, generate_kwargs={"max_new_tokens": 100})
 
                
                prompt = outputs[0]["generated_text"]
                prompt = prompt[82:-1]
 
 
                #imagegen = pipe(prompt, cropped_img_label, num_inference_steps=35).images[0]
 
                image = pipe(
                           prompt=prompt,
                           image=img_label,
                           ip_adapter_image=ip_adapter_image,
                           negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                           num_inference_steps=50,
                          ).images[0]
                image.save('nn.png')
 
                #seg
                inputs = processor(images=image, return_tensors="pt") 
                with torch.no_grad():
                     outputs = model(**inputs)
                
                predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
                
                ##
                predicted_semantic_map=predicted_semantic_map.numpy()
                match = np.sum(predicted_semantic_map == gtimage)
                total = predicted_semantic_map.size
                ratio = match / total
                logging.info(f"ratio:{ratio}")
                #print('ratio',ratio)
                 
 

                save_gen = os.path.join(root, f"{file}")
 
                save_gen = save_gen.replace("color.png", "color_syc.png")
                save_gen = save_gen.replace("gtFine_trainvaltest/gtFine/train", "syc_i2t")
                 
                image.save(save_gen)
                
                logging.info(f"save_gen:{save_gen}")
                #print(save_gen)
 
 
