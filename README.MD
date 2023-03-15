# MonoDetector
This repository contains unofficial pytorch implementations for **[MonoCon: Learning Auxiliary Monocular Contexts Helps Monocular 3D Object Detection (AAAI, 2022)](https://arxiv.org/abs/2112.04628)**, **[IDMS: Instance Depth for Multi-scale Monocular 3D Object Detection (JMLR, 2022)](https://arxiv.org/abs/2212.01528)**, **[SMOKE: Single-Stage Monocular 3D Object Detection via Keypoint Estimation (CVPRW, 2020)](https://arxiv.org/pdf/2002.10111.pdf)**, **[(MonoFlex) Objects are Different: Flexible Monocular 3D Object Detection (CVPR, 2021)](https://arxiv.org/abs/2104.02323)** and **[(GUPNet) Geometry Uncertainty Projection Network for Monocular 3D Object Detection (ICCV, 2021)](https://arxiv.org/abs/2107.13774)**

<img src="resources/demo_3d_video.gif">

## Features
Unchecked items are currently being prepared.
- [x] Support various mono3D detectors
- [x] KITTI Evaluation
- [x] Visualization (2D Bounding Box + Projected 3D Bounding Box)
- [x] Visualization (Bird Eye's View)
- [x] Video Inference using KITTI Raw Data Sequences
- [x] Multi-GPU Training
- [ ] FPN


## Quantitative Results

This repov3 includes instance depth head and ddp mode
### 3D Metric on Car Class
|                | AP40@Easy     | AP40@Mod.     | AP40@Hard     |
| -------------- | ---------     |-----------    |-----------    |
| Smoke(paper)   | 14.03         |  9.76         |  7.84         |
| This Repo      | 18.62 (+4.59) | 14.24 (+4.48) | 12.72 (+4.88) | 
| This Repo v2   | 20.45 (+6.42) | 15.58 (+5.82) | 13.20 (+5.36) |
| This Repo v3   | 21.24 (+7.21) | 16.00 (+6.24) | 13.47 (+5.63) | 
| MonoFlex(paper)| 23.64         | 17.51         | 14.83         |
| This Repo      | 21.68 (-1.96) | 15.64 (-1.87) | 13.19 (-1.64) |
| GUPNet(paper)  | 22.76         | 16.46         | 13.72         |  
| This Repo      | 20.28 (-2.48) | 14.39 (-2.07) | 12.78 (-0.94) |
| MonoCon(paper) | 26.33         | 19.03         | 16.00         |
| This Repo      | 26.03 (-0.30) | 19.02 (-0.01) | 15.92 (-0.08) | 
| IDMS(paper)    | 22.59         | 17.79         | 14.99         |
| This Repo      | 20.40 (-2.19) | 15.33 (-2.45) | 12.94 (-2.05) | 

  

## Preparations

### Setup
```bash
# [Step 1]: Create new conda environment and activate.
#           Set [ENV_NAME] freely to any name you want. (Please exclude the brackets.)
conda create --name [ENV_NAME] python=3.9
conda activate [ENV_NAME]

# [Step 2]: Clone this repository and change directory.
git clone https://github.com/zhangcheng828/MonoDetector.git
cd monodetector

# [Step 3]: See https://pytorch.org/get-started/locally/ and install pytorch for your environment.
#           We have tested on version 1.11.0.
#           It is recommended to install version 1.7.0 or higher.

# [Step 4]: Install some packages using 'requirements.txt' in the repository.
pip install -r requirements.txt
```

### Environments
We have tested our code in the following environment.  
| OS                 | Python       | Pytorch      | CUDA         | GPU                   
| :----------------: | :----------: | :----------: | :----------: | :-------------------: 
| Ubuntu 18.04.5 LTS | 3.9.15       | 1.11.0       | 11.1         | NVIDIA RTX 3090       



## Dataset
### KITTI 3D Object Detection Dataset
Please download **KITTI dataset** from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).  
Click the three items below from the link to download it and unzip it into the same folder.
- Download left color images of object data set (12 GB)
- Download camera calibration matrices of object data set (16 MB)
- Download training labels of object data set (5 MB)

The structure of the data files should be as below.
```bash
[ROOT]
│
├── training
│   ├── calib
│   │   ├── 000000.txt
│   │   ├── 000001.txt
│   │   └── ...
│   ├── image_2
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   └── label_2
│       ├── 000000.txt
│       ├── 000001.txt
│       └── ...
│
└── testing
    ├── calib
    ├── image_2
    └── label_2
```


## Usage
### Training
Just edit the items in ```config/default.py``` before execution.  
If your GPU memory is **less than 16GB**, please set ```_C.USE_BENCHMARK``` in ```config/default.py``` to ```False```.
```bash
python tools/train.py config/smoke.yaml
```
There is currently a bug, using local_rank 0 for model evaluation will cause other process to wait too long and report errors, so the default EVAL_PERIOD is set to -1.
```bash
torchrun --nproc_per_node=4 tools/train_ddp.py config/smoke.yaml # ddp mode
```

### Evaluation
```bash
python tools/test.py  --config_file       [FILL]      # Config file (.yaml file)
                --checkpoint_file   [FILL]      # Checkpoint file (.pth file)
python tools/test.py --config_file=config/monocon.yaml --checkpoint_file=logs/monocon_12_08/checkpoints/epoch_198.pth

torchrun --nproc_per_node=1 tools/train_ddp.py config/smoke.yaml --eval=True # ddp mode
```

### Inference
```bash
python tools/test.py  --config_file       [FILL]      # Config file (.yaml file)
                --checkpoint_file   [FILL]      # Checkpoint file (.pth file)
                --visualize                     # Perform visualization (Qualitative Results)
                --gpu_id            [Optional]  # Index of GPU to use for testing (Default: 0)
                --save_dir          [FILL]      # Path where visualization results will be saved to
```


### Video Inference on KITTI Raw Dataset
The KITTI Raw Dataset can be downloaded by scene from [here](https://www.cvlibs.net/datasets/kitti/raw_data.php?type=city). (You will probably need to log in.)  
There are 4 download links for each scene, of which, please download only **"synced+rectified data"** and **"calibration"**.
```bash
python tools/test_raw.py  --data_dir          [FILL]      # Path where sequence images are saved
                    --calib_file        [FILL]      # Calibration file ("calib_cam_to_cam.txt")
                    --checkpoint_file   [FILL]      # Checkpoint file (.pth file)
                    --gpu_id            [Optional]  # Index of GPU to use for testing (Default: 0)
                    --fps               [Optional]  # FPS of the result video (Default: 25)
                    --save_dir          [FILL]      # Path of the directory to save the result video
```

## Qualitative Results  
Visualizations for 2D Boxes, 3D Boxes, and BEV, respectively, from top to bottom for each sample.  
- **000008.png** (Validation Set)  
<img src="resources/000008_2d.png">
<img src="resources/000008_3d.png">  
<img src="resources/000008_bev.png">  

- **000134.png** (Validation Set)  
<img src="resources/000134_2d.png">
<img src="resources/000134_3d.png">  
<img src="resources/000134_bev.png">  

## Change Log
This repository was last updated to **v0.03** on **2022.12.14**.  


## References
The following repositories were referred.
- [Smoke](https://github.com/lzccccc/SMOKE/tree/master/smoke)  
- [MonoFlex](https://github.com/zhangyp15/MonoFlex)  
- [GUPNet](https://github.com/SuperMHP/GUPNet)  
- [MonoCon-pytorch](https://github.com/2gunsu/monocon-pytorch)  
- [MonoCon](https://github.com/Xianpeng919/MonoCon)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit)
