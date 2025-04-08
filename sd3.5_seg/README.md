1 环境配置

```conda create -n name python==3.10.8```

```pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121```

```pip install -r requirments.txt```

安装cityscapesScripts和detectron2

python -m pip install cityscapesscripts 

git clone https://github.com/facebookresearch/detectron2.git

python -m pip install -e detectron2

2 数据库

下载https://www.cityscapes-dataset.com/downloads/

gtFine_trainvaltest.zip  https://www.cityscapes-dataset.com/file-handling/?packageID=1

leftImg8bit_trainvaltest.zip  https://www.cityscapes-dataset.com/file-handling/?packageID=3

数据库目录
> data
>> cityscapes
>>> gtFine
>>>>> test  
>>>>> train  
>>>>> val  

>>> leftImg8bit
>>>>> test  
>>>>> train  
>>>>> val  
 
数据库预处理（转化训练ID，参考https://github.com/facebookresearch/Mask2Former/tree/main/datasets ）

```CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createTrainIdLabelImgs.py```


3 models下载（参考https://github.com/Stability-AI/sd3.5?tab=readme-ov-file#download ）

> models
>> clip_g.safetensors  
>> clip_l.safetensors  
>> sd3.5_medium.safetensors  
>> t5xxl.safetensors  
>> t5xxl_fp8_e4m3fn.safetensors  
 

4 运行文件
```sh train_semantic_Cityscapes2.sh```



5 参考
sd3.5 + DatasetDM  
https://github.com/Stability-AI/sd3.5

https://github.com/showlab/DatasetDM

