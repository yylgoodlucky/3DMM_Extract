## Environment

```
git clone https://github.com/yylgoodlucky/3DMM_Extract.git
cd 3DMM_Extract
conda create -n 3DMM_Extract python=3.7
conda activate 3DMM_Extract
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

## Quick Start

#### Pretrained Models

Put pretrain model :https://drive.google.com/drive/folders/1U7qi2hyNEWqPgCmsS0brJUJtsQj4v_ZZ input rootdir.
|checkpoints/BFM | 3DMM library. (Note the zip file should be unzipped to BFM/.)
|checkpoints/Deep3D/epoch_20.pth | Pre-trained 3DMM extractor.

We extract corresponding 3dmm parameters with video or image

#### Inference

```
python main.py \
 --video_path ./video_root 


python main.py \
 --image_path ./image_root
```


The `--video_path` and `--image_path` can be specified as either a single file or a folder.

If you need align (crop) images during the inference process, please specify `--if_align`. 

You can first extract the 3dmm parameters with the script `TODO.sh` and save the 3dmm in the `{video_source}/3dmm/3dmm_{video_name}.npy`

The 3dmm parameters of the images can also be pre-extracted or online-extracted with the parameter `--if_extract`.


## 🥂 Related Works
- [SadTalker： Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023)](https://github.com/Winfredy/SadTalker)
- [CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior (CVPR 2023)](https://github.com/Doubiiu/CodeTalker)
- [VideoReTalking: Audio-based Lip Synchronization for Talking Head Video Editing In the Wild (SIGGRAPH Asia 2022)](https://github.com/vinthony/video-retalking)
- [DPE: Disentanglement of Pose and Expression for General Video Portrait Editing (CVPR 2023)](https://github.com/Carlyx/DPE)
- [3D GAN Inversion with Facial Symmetry Prior (CVPR 2023)](https://github.com/FeiiYin/SPI/)
- [T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations (CVPR 2023)](https://github.com/Mael-zys/T2M-GPT)


