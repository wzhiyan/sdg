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

logging.basicConfig(filename="logfile.log", level = logging.INFO)

 
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
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


label2trainid   = { label.id      : label.trainId for label in labels   }
trainId2name   = { label.trainId : label.name for label in labels   }
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
#directory = "/home/lenovo/wzy/contseg/cityscapes/try" 
directory = "/home/lenovo/wzy/contseg/cityscapes/gtFine_trainvaltest/gtFine/train" 

new = "/home/lenovo/wzy/contseg/cityscapes/new"  

for root, dirs, files in os.walk(directory):
        for file in files:    
            if '_gtFine_color.png' in file:               
                image_path_label = os.path.join(root, file)
                image_path_gt = image_path_label.replace("gtFine_color.png", "gtFine_labelIds.png") 
                if os.path.exists(image_path_gt):
                        with Image.open(image_path_label) as img_label:
                            width, height = img_label.size                            
                            if width >= 768 and height >= 768:
                                left = (width - 768) / 2
                                upper = (height - 768) / 2
                                right = (width + 768) / 2
                                lower = (height + 768) / 2  
                                img_label=img_label.convert('RGB')                              
                                cropped_img_label = img_label.crop((left, upper, right, lower))                                
                                save_path_label1 = os.path.join(root, f"cropped_{file}")
                                save_path_label = save_path_label1.replace("gtFine_trainvaltest/gtFine/train", "new")
                                cropped_img_label.save(save_path_label)
                        with Image.open(image_path_gt) as img_gt:
                            width, height = img_gt.size                            
                            if width >= 768 and height >= 768:
                                left = (width - 768) / 2
                                upper = (height - 768) / 2
                                right = (width + 768) / 2
                                lower = (height + 768) / 2
                                
                                cropped_img_gt = img_gt.crop((left, upper, right, lower)).convert("L")
                                save_path_gt1 = os.path.join(root, f"cropped_{file.replace('gtFine_color.png', 'gtFine_labelIds.png')}")
                                save_path_gt = save_path_gt1.replace("gtFine_trainvaltest/gtFine/train", "new")
                                cropped_img_gt.save(save_path_gt)                                
                                gtimage = np.array(cropped_img_gt)
                                 

                                for k, v in label2trainid.items():
                                    gtimage[gtimage == k] = v
                                    
                                ids=np.unique(gtimage)
                                
                                
                                ids_s=np.unique(ids)
                                ids_s=ids_s[:-1]
                                output = ""
                                for k, v in trainId2name.items():
                                  if k in ids_s:
                                     output = output + "," + v

             
                prompt="street scene" + output
                #print('prompt',prompt)
                #image = Image.fromarray(cropped_img_label)
 
                imagegen = pipe(prompt, cropped_img_label, num_inference_steps=35).images[0]
 
                #seg
                inputs = processor(images=imagegen, return_tensors="pt") 
                with torch.no_grad():
                     outputs = model(**inputs)
                
                predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[imagegen.size[::-1]])[0]
                
                ##
                predicted_semantic_map=predicted_semantic_map.numpy()
                match = np.sum(predicted_semantic_map == gtimage)
                total = predicted_semantic_map.size
                ratio = match / total
                logging.info(f"ratio:{ratio}")
                #print('ratio',ratio)

                save_gen = os.path.join(root, f"{file}")
                save_gen = save_gen.replace("color.png", "color_syc.png")
                save_gen = save_gen.replace("gtFine_trainvaltest/gtFine/train", "new/syc")
                imagegen.save(save_gen)
                logging.info(f"save_gen:{save_gen}")
                #print(save_gen)
                
                 
                
                #print(sss)
                
 
