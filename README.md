English | [简体中文](README_cn.md)

<div align="center"><img src="assets/logo.png" width="1500"></div>

## Introduction
Welcome to **DAMO-YOLO**! It is a fast and accurate object detection method, which is developed by TinyML Team from Alibaba DAMO Data Analytics and Intelligence Lab. And it achieves a higher performance than state-of-the-art YOLO series. DAMO-YOLO is extend from YOLO but with some new techs, including Neural Architecture Search (NAS) backbones, efficient Reparameterized Generalized-FPN (RepGFPN), a lightweight head with AlignedOTA label assignment, and distillation enhancement. For more details, please refer to our [Arxiv Report](https://arxiv.org/abs/2211.15444). Moreover, here you can find not only powerful models, but also highly efficient training strategies and complete tools from training to deployment.

<div align="center"><img src="assets/curve.png" width="500"></div>

## Updates
- **[2022/12/15: We release  DAMO-YOLO v0.1.1!]**
  * Add a detailed [Custom Dataset Finetune Tutorial](./assets/CustomDatasetTutorial.md).
- **[2022/11/27: We release  DAMO-YOLO v0.1.0!]**
  * Release DAMO-YOLO object detection models, including DAMO-YOLO-T, DAMO-YOLO-S and DAMO-YOLO-M.
  * Release model convert tools for easy deployment, supports onnx and TensorRT-fp32, TensorRT-fp16.

## Web Demo
- [DAMO-YOLO-T](https://www.modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo-t/summary), [DAMO-YOLO-S](https://modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo/summary), [DAMO-YOLO-M](https://www.modelscope.cn/models/damo/cv_tinynas_object-detection_damoyolo-m/summary) is integrated into ModelScope. Try out the Web Demo.



###

```
pip install damo_yolo_detect
```


```
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
object_detect = pipeline(Tasks.image_object_detection,model='damo/cv_tinynas_object-detection_damoyolo')
img_path ='https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_detection.jpg'
result = object_detect(img_path)

```

## Model Zoo
|Model |size |mAP<sup>val<br>0.5:0.95 | Latency T4<br>TRT-FP16-BS1| FLOPs<br>(G)| Params<br>(M)| Download |
| ------        |:---: | :---:     |:---:|:---: | :---: | :---:|
|[DAMO-YOLO-T](./configs/damoyolo_tinynasL20_T.py) | 640 | 41.8  | 2.78  | 18.1  | 8.5  |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/before_distill/damoyolo_tinynasL20_T_418.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/before_distill/damoyolo_tinynasL20_T_418.onnx)  |
|[DAMO-YOLO-T*](./configs/damoyolo_tinynasL20_T.py) | 640 | 43.0  | 2.78  | 18.1  | 8.5  |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/damoyolo_tinynasL20_T.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/damoyolo_tinynasL20_T.onnx)  |
|[DAMO-YOLO-S](./configs/damoyolo_tinynasL25_S.py) | 640 | 45.6  | 3.83  | 37.8  | 16.3 |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/before_distill/damoyolo_tinynasL25_S_456.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/before_distill/damoyolo_tinynasL25_S_456.onnx)  |
|[DAMO-YOLO-S*](./configs/damoyolo_tinynasL25_S.py) | 640 | 46.8  | 3.83  | 37.8  | 16.3 |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/damoyolo_tinynasL25_S.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/damoyolo_tinynasL25_S.onnx)  |
|[DAMO-YOLO-M](./configs/damoyolo_tinynasL35_M.py) | 640 | 48.7  | 5.62  | 61.8  | 28.2 |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/before_distill/damoyolo_tinynasL35_M_487.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/before_distill/damoyolo_tinynasL35_M_487.onnx)|
|[DAMO-YOLO-M*](./configs/damoyolo_tinynasL35_M.py) | 640 | 50.0  | 5.62  | 61.8  | 28.2 |[torch](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/clean_models/damoyolo_tinynasL35_M.pth),[onnx](https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/onnx/damoyolo_tinynasL35_M.onnx)|

- We report the mAP of models on COCO2017 validation set, with multi-class NMS.
- The latency in this table is measured without post-processing.
- \* denotes the model trained with distillation.

## Quick Start

<details>
<summary>Installation</summary>

Step1. Install DAMO-YOLO.
```shell
git clone https://github.com/tinyvision/damo-yolo.git
cd DAMO-YOLO/
conda create -n DAMO-YOLO python=3.7 -y
conda activate DAMO-YOLO
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```
Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython;
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
</details>

<details>
<summary>Demo</summary>

Step1. Download a pretrained torch model or onnx engine from the benchmark table, e.g., damoyolo_tinynasL25_S.pth or damoyolo_tinynasL25_S.onnx.

Step2. Use -f(config filename) to specify your detector's config. For example:
```shell
# torch
python tools/torch_inference.py -f configs/damoyolo_tinynasL25_S.py --ckpt /path/to/your/damoyolo_tinynasL25_S.pth --path assets/dog.jpg

# onnx
python tools/onnx_inference.py -f configs/damoyolo_tinynasL25_S.py --onnx /path/to/your/damoyolo_tinynasL25_S.onnx --path assets/dog.jpg
```

</details>


<details>
<summary>Reproduce our results on COCO</summary>

Step1. Prepare COCO dataset
```shell
cd <DAMO-YOLO Home>
ln -s /path/to/your/coco ./datasets/coco
```

Step 2. Reproduce our results on COCO by specifying -f(config filename)
```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/train.py -f configs/damoyolo_tinynasL25_S.py
```
</details>

<details>
<summary>Finetune on your data</summary>

Please refer to [custom dataset tutorial](./assets/CustomDatasetTutorial.md) for details.

</details>



<details>
<summary>Evaluation</summary>

```shell
python -m torch.distributed.launch --nproc_per_node=8 tools/eval.py -f configs/damoyolo_tinynasL25_S.py --ckpt /path/to/your/damoyolo_tinynasL25_S.pth
```
</details>


<details>
<summary>Customize tinynas backbone</summary>
Step1. If you want to customize your own backbone, please refer to [MAE-NAS Tutorial for DAMO-YOLO](https://github.com/alibaba/lightweight-neural-architecture-search/blob/main/scripts/damo-yolo/Tutorial_NAS_for_DAMO-YOLO_cn.md). This is a detailed tutorial about how to obtain an optimal backbone under the budget of latency/flops.  

Step2. After the searching process completed, you can replace the structure text in configs with it. Finally, you can get your own custom ResNet-like or CSPNet-like backbone after setting the backbone name to TinyNAS_res or TinyNAS_csp. Please notice the difference of out_indices between TinyNAS_res and TinyNAS_csp. 
```
structure = self.read_structure('tinynas_customize.txt')
TinyNAS = { 'name'='TinyNAS_res', # ResNet-like Tinynas backbone
            'out_indices': (2,4,5)}
TinyNAS = { 'name'='TinyNAS_csp', # CSPNet-like Tinynas backbone
            'out_indices': (2,3,4)}

```
</details>



## Deploy
<details>
<summary>Installation</summary>

Step1. Install ONNX.
```shell
pip install onnx==1.8.1
pip install onnxruntime==1.8.0
pip install onnx-simplifier==0.3.5
```
Step2. Install CUDA、CuDNN、TensorRT and pyCUDA

2.1 CUDA
```shell
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run
export PATH=$PATH:/usr/local/cuda-10.2/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
source ~/.bashrc
```
2.2 CuDNN
```shell
sudo cp cuda/include/* /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```
2.3 TensorRT
```shell
cd TensorRT-7.2.1.6/python
pip install tensorrt-7.2.1.6-cp37-none-linux_x86_64.whl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TensorRT-7.2.1.6/lib
```
2.4 pycuda
```shell
pip install pycuda==2022.1
```
</details>

<details>
<summary>Model Convert</summary>

Step.1 convert torch model to onnx or trt engine, and the output file would be generated in ./deploy. end2end means to export trt with nms. trt_eval means to evaluate the exported trt engine on coco_val dataset after the export compelete.
```shell
# onnx export 
python tools/converter.py -f configs/damoyolo_tinynasL25_S.py -c damoyolo_tinynasL25_S.pth --batch_size 1 --img_size 640

# trt export
python tools/converter.py -f configs/damoyolo_tinynasL25_S.py -c damoyolo_tinynasL25_S.pth --batch_size 1 --img_size 640 --trt --end2end --trt_eval
```

Step.2 trt engine evaluation on coco_val dataset. end2end means to using trt_with_nms to evaluation.
```shell
python tools/trt_eval.py -f configs/damoyolo_tinynasL25_S.py -trt deploy/damoyolo_tinynasL25_S_end2end.trt --batch_size 1 --img_size 640 --end2end
```

Step.3 onnx or trt engine inference demo and appoint test image by -p. end2end means to using trt_with_nms to inference.
```shell
# onnx inference
python tools/onnx_inference.py -f configs/damoyolo_tinynasL25_S.py --onnx /path/to/your/damoyolo_tinynasL25_S.onnx --path assets/dog.jpg

# trt inference
python tools/trt_inference.py -f configs/damoyolo_tinynasL25_S.py -t deploy/damoyolo_tinynasL25_S_end2end_fp16_bs1.trt -p assets/dog.jpg --img_size 640 --end2end
```
</details>

## Intern Recruitment
We are recruiting research intern, if you are interested in object detection, model quantization or NAS, please send your resume to xiuyu.sxy@alibaba-inc.com  


## Cite DAMO-YOLO
If you use DAMO-YOLO in your research, please cite our work by using the following BibTeX entry:

```latex
 @article{damoyolo,
   title={DAMO-YOLO: A Report on Real-Time Object Detection Design},
   author={Xianzhe Xu, Yiqi Jiang, Weihua Chen, Yilun Huang, Yuan Zhang and Xiuyu Sun},
   journal={arXiv preprint arXiv:2211.15444},
   year={2022},
 }
```
