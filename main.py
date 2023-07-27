import os
import argparse
import numpy as np
import torch
import pdb
import utils.video_util as video_util

from os.path import join
from tqdm import tqdm
from torchvision import transforms
from utils.preprocess.align_face import align_image, align_image_pil
from utils.preprocess.extract_landmark import get_landmark
from utils.preprocess.crop_videos_inference import Croper
from utils.preprocess.extract_3dmm import Extract3dmm
from PIL import Image

class Extractor:
    def __init__(self, model_3dmm, if_align=False, resize=256):
        self.model_3dmm = model_3dmm
        self.if_align = if_align
        # if len(self.video_list) != len(self.image_list) and len(self.video_list) > 0:
        #     self.video_list = self.video_list * (len(self.image_list) // len(self.video_list) + 1)
        #     self.video_list = self.video_list[:len(self.image_list)]

        self.video_index = -1
        self.transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ])
        self.croper = Croper()
        
    def get_param(self, video_path=None, image_path=None):
        # Hard code; Bad writing
        if video_path is not None:
            video_name = os.path.basename(video_path).split('.')[0]
            frames_pil = video_util.read_video(video_path, resize=256)

            save_3dmm_path = os.path.join(os.path.dirname(video_path), '3dmm', '3dmm_' + video_name + '.npy')
            if os.path.exists(save_3dmm_path):
                os.makedirs(os.path.join(os.path.dirname(video_path), '3dmm'), exist_ok=True)
                lm_np = get_landmark(frames_pil)

                frames_pil = self.croper.crop(frames_pil, lm_np)
                lm_np = get_landmark(frames_pil)

                coeff_3dmm = self.model_3dmm.get_3dmm(frames_pil, lm_np)
                # print(coeff_3dmm.shape) # (252, 73)
                np.save(save_3dmm_path, coeff_3dmm)

            coeff_3dmm = np.load(save_3dmm_path, allow_pickle=True)
            coeff_3dmm = torch.from_numpy(coeff_3dmm)

        if image_path is not None:
            src_image_pil = Image.open(image_path).convert("RGB")  # prevent png exist channel error

            image_name = os.path.basename(image_path).split('.')[0]
            source_3dmm_path = os.path.join(os.path.dirname(image_path), '3dmm', '3dmm_' + image_name + '.npy')
            if not os.path.exists(source_3dmm_path):
                src_image_pil_256 = src_image_pil.resize((256, 256))
                os.makedirs(os.path.join(os.path.dirname(image_path), '3dmm'), exist_ok=True)
                lm_np = get_landmark([src_image_pil_256])
                # print(lm_np.shape)
                source_3dmm = self.model_3dmm.get_3dmm([src_image_pil_256], lm_np)
                # print(coeff_3dmm.shape)
                np.save(source_3dmm_path, source_3dmm)

            source_3dmm = np.load(source_3dmm_path, allow_pickle=True)
            source_3dmm = torch.from_numpy(source_3dmm)
        else:
            src_image_pil = frames_pil[0]
            source_3dmm = coeff_3dmm[0].unsqueeze(0)

        if self.if_align:
            src_lm_np = get_landmark([src_image_pil])
            src_align_pil = align_image_pil([src_image_pil], src_lm_np)
            src_align_pil = src_align_pil[0]
        else:
            src_align_pil = src_image_pil

        return {
            'source_align': src_align_pil,
            'source_image': src_image_pil,
            'source_3dmm': source_3dmm,
            'frames': frames_pil if video_path is not None else None,
            'coeff_3dmm': coeff_3dmm if video_path is not None else None,
            'video_name': video_name if video_path is not None else None
        }
    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument('--if_align', action='store_true')
    
    args = parser.parse_args()

    extractor = Extractor(model_3dmm=Extract3dmm(), if_align=args.if_align, resize=1024)
    
    if args.video_path:
        video_list = [join(args.video_path, file) for file in sorted(os.listdir(args.video_path)) if file.endswith('.mp4')]

        for video in tqdm(video_list):
            extractor.get_param(video_path=video)
            
    if args.image_path:
        image_list = [join(args.image_path, file) for file in sorted(os.listdir(args.image_path)) if file.endswith('.png') or file.endswith('.jpg')]

        for image in tqdm(image_list):
            extractor.get_param(image_path=image)