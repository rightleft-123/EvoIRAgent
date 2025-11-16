import argparse
import json
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from glob import glob

from io import BytesIO
import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path
from torch.utils import data as data
from torchvision.transforms.functional import gaussian_blur
import matplotlib.pyplot as plt

from basicsr.utils import DiffJPEG
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels

from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor, tensor2img
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register(suffix='basicsr')
class PairedDataset(data.Dataset):
    """Modified dataset based on the dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(PairedDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt.io_backend
        self.dataroot_gt = opt.dataroot_gt
        self.process_type = opt.process_type

        if 'crop_size' in opt:
            self.crop_size = opt.crop_size
        else:
            self.crop_size = 512
        if 'image_type' not in opt:
            opt.image_type = 'png'

        # support multiple type of data: file path and meta data, remove support of lmdb
        self.process_data_num = opt.process_data_num
        # if opt.process_type == "data_preprocessing":
        #     assert self.process_data_num <= 350, "The number of data should be less than 350"

        tmp_lq_paths, tmp_gt_paths = [], []
        self.lq_paths = []
        self.gt_paths = []
        self.gt_degradation_types = []
        if 'blurring_path' in opt:
            if isinstance(opt['blurring_path'], str):
                # Use rglob to recursively search for images
                for j, doc in enumerate(Path(os.path.join(self.dataroot_gt, opt['blurring_path'])).rglob("*")):
                    if doc.is_dir():
                        tmp_lq_paths.extend(sorted([str(x) for x in Path(doc, "blur").rglob('*.' + opt['image_type'])]))
                        tmp_gt_paths.extend(sorted([str(x) for x in Path(doc, "sharp").rglob('*.' + opt['image_type'])]))
                    else:
                        break
                assert self.process_data_num <= len(tmp_lq_paths), f"The number of data should be less than the number of data in the folder={len(tmp_lq_paths)}"

                paired_paths = random.sample(list(zip(tmp_lq_paths, tmp_gt_paths)), self.process_data_num)
                tmp_lq_paths, tmp_gt_paths = zip(*paired_paths)
                self.lq_paths.extend(list(tmp_lq_paths))
                self.gt_paths.extend(list(tmp_gt_paths))
                tmp_lq_paths = []
                tmp_gt_paths = []
        if 'hazy_path' in opt:
            if isinstance(opt['hazy_path'], str) and opt['hazy_path'] != ' ':
                # Use rglob to recursively search for images
                tmp_lq_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['hazy_path']), "hazy").rglob('*.' + 'jpg')]))
                tmp_lq_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['hazy_path']), "JPEGImages").rglob('*.' + 'png')]))
                tmp_gt_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['hazy_path']), "GT").rglob('*.' + 'jpg')]))
                tmp_gt_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['hazy_path']), "JPEGImages").rglob('*.' + 'png')]))
                assert self.process_data_num <= len(tmp_lq_paths), f"The number of data should be less than the number of data in the folder={len(tmp_lq_paths)}"

                paired_paths = random.sample(list(zip(tmp_lq_paths, tmp_gt_paths)), self.process_data_num)
                tmp_lq_paths, tmp_gt_paths = zip(*paired_paths)
                self.lq_paths.extend(list(tmp_lq_paths))
                self.gt_paths.extend(list(tmp_gt_paths))
                self.gt_degradation_types.extend(["haze"] * len(tmp_lq_paths))  # 假设hazy图像的类别标签为0
                tmp_lq_paths = []
                tmp_gt_paths = []
        if 'low_light_path' in opt:
            if isinstance(opt['low_light_path'], str):
                # Use rglob to recursively search for images
                tmp_lq_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['low_light_path']), "low").rglob('*.' + opt['image_type'])]))
                tmp_gt_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['low_light_path']), "high").rglob('*.' + opt['image_type'])]))
                assert self.process_data_num <= len(tmp_lq_paths), f"The number of data should be less than the number of data in the folder={len(tmp_lq_paths)}"
          
                paired_paths = random.sample(list(zip(tmp_lq_paths, tmp_gt_paths)), self.process_data_num)
                tmp_lq_paths, tmp_gt_paths = zip(*paired_paths)
                self.lq_paths.extend(list(tmp_lq_paths))
                self.gt_paths.extend(list(tmp_gt_paths))
                tmp_lq_paths = []
                tmp_gt_paths = []
        if 'raindrop_path' in opt:
            if isinstance(opt['raindrop_path'], str) and opt['raindrop_path'] != ' ':
                # Use rglob to recursively search for images
                tmp_lq_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['raindrop_path']), "data").rglob('*.' + opt['image_type'])]))
                tmp_gt_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['raindrop_path']), "gt").rglob('*.' + opt['image_type'])]))

                if self.process_data_num == 4000:
                    # Create 5 copies of 800 samples each (total 4000)
                    base_sample_size = 800
                    all_paired_paths = []
                    for _ in range(5):
                        paired_paths = random.sample(list(zip(tmp_lq_paths, tmp_gt_paths)), base_sample_size)
                        all_paired_paths.extend(paired_paths)
                    paired_paths = all_paired_paths
                    tmp_lq_paths, tmp_gt_paths = zip(*paired_paths)
                # assert self.process_data_num <= len(tmp_lq_paths), f"The number of data should be less than the number of data in the folder={len(tmp_lq_paths)}"
                else:
                    paired_paths = random.sample(list(zip(tmp_lq_paths, tmp_gt_paths)), self.process_data_num)
                    tmp_lq_paths, tmp_gt_paths = zip(*paired_paths)
                self.lq_paths.extend(list(tmp_lq_paths))
                self.gt_paths.extend(list(tmp_gt_paths))
                self.gt_degradation_types.extend([torch.tensor([3])] * len(tmp_lq_paths))  # 假设raindrop图像的类别标签为1
                tmp_lq_paths = []
                tmp_gt_paths = []
        if 'shadowed_path' in opt:
            if isinstance(opt['shadowed_path'], str):
                # Use rglob to recursively search for images
                tmp_lq_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['shadowed_path']), "shadow").rglob('*.' + "jpg")]))
                tmp_gt_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['shadowed_path']), "shadow_free").rglob('*.' + "jpg")]))
                assert self.process_data_num <= len(tmp_lq_paths), f"The number of data should be less than the number of data in the folder={len(tmp_lq_paths)}"
           
                paired_paths = random.sample(list(zip(tmp_lq_paths, tmp_gt_paths)), self.process_data_num)
                tmp_lq_paths, tmp_gt_paths = zip(*paired_paths)
                self.lq_paths.extend(list(tmp_lq_paths))
                self.gt_paths.extend(list(tmp_gt_paths))
                tmp_lq_paths = []
                tmp_gt_paths = []
        if 'snowy_path' in opt:
            if isinstance(opt['snowy_path'], str) and opt['snowy_path'] != ' ':
                # Use rglob to recursively search for images
                tmp_lq_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['snowy_path'], 'video2imgs_IN_testing_re')).rglob('*.' + "jpg")]))
                tmp_gt_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['snowy_path'], 'video2imgs_GT_testing_re')).rglob('*.' + "jpg")]))
                assert self.process_data_num <= len(tmp_lq_paths), f"The number of data should be less than the number of data in the folder={len(tmp_lq_paths)}"
          
                paired_paths = random.sample(list(zip(tmp_lq_paths, tmp_gt_paths)), self.process_data_num)
                tmp_lq_paths, tmp_gt_paths = zip(*paired_paths)
                self.lq_paths.extend(list(tmp_lq_paths))
                self.gt_paths.extend(list(tmp_gt_paths))
                self.gt_degradation_types.extend([torch.tensor([0])] * len(tmp_lq_paths))  # 假设snowy图像的类别标签为2
                tmp_lq_paths = []
                tmp_gt_paths = []
        if 'jpeg_path' in opt:
            if isinstance(opt['jpeg_path'], str):
                # Use rglob to recursively search for images
                tmp_lq_paths.extend(sorted(["jpeg compression artifact" for _ in Path(os.path.join(self.dataroot_gt, opt['jpeg_path'])).rglob('*.' + "png")]))
                tmp_gt_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['jpeg_path'])).rglob('*.' + "png")]))
                assert self.process_data_num <= len(tmp_lq_paths), f"The number of data should be less than the number of data in the folder={len(tmp_lq_paths)}"
         
                paired_paths = random.sample(list(zip(tmp_lq_paths, tmp_gt_paths)), self.process_data_num)
                tmp_lq_paths, tmp_gt_paths = zip(*paired_paths)
                self.lq_paths.extend(list(tmp_lq_paths))
                self.gt_paths.extend(list(tmp_gt_paths))
                self.gt_degradation_types.extend(["jpeg compression artifact"] * len(tmp_lq_paths))
                tmp_lq_paths = []
                tmp_gt_paths = []
        if 'ringing_path' in opt:
            if isinstance(opt['ringing_path'], str):
                # Use rglob to recursively search for images
                tmp_lq_paths.extend(sorted(["ringing" for x in Path(os.path.join(self.dataroot_gt, opt['ringing_path'])).rglob('*.' + "png")]))
                tmp_gt_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['ringing_path'])).rglob('*.' + "png")]))
                assert self.process_data_num <= len(tmp_lq_paths), f"The number of data should be less than the number of data in the folder={len(tmp_lq_paths)}"
          
                paired_paths = random.sample(list(zip(tmp_lq_paths, tmp_gt_paths)), self.process_data_num)
                tmp_lq_paths, tmp_gt_paths = zip(*paired_paths)
                self.lq_paths.extend(list(tmp_lq_paths))
                self.gt_paths.extend(list(tmp_gt_paths))
                tmp_lq_paths = []
                tmp_gt_paths = []
        if 'poisson_path' in opt:
            if isinstance(opt['poisson_path'], str):
                # Use rglob to recursively search for images
                tmp_lq_paths.extend(sorted(['poisson' for x in Path(os.path.join(self.dataroot_gt, opt['poisson_path'])).rglob('*.' + "png")]))
                tmp_gt_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['poisson_path'])).rglob('*.' + "png")]))
                assert self.process_data_num <= len(tmp_lq_paths), f"The number of data should be less than the number of data in the folder={len(tmp_lq_paths)}"
            
                paired_paths = random.sample(list(zip(tmp_lq_paths, tmp_gt_paths)), self.process_data_num)
                tmp_lq_paths, tmp_gt_paths = zip(*paired_paths)
                self.lq_paths.extend(list(tmp_lq_paths))
                self.gt_paths.extend(list(tmp_gt_paths))
                tmp_lq_paths = []
                tmp_gt_paths = []
        if 'saltpepper_path' in opt:
            if isinstance(opt['saltpepper_path'], str):
                # Use rglob to recursively search for images
                tmp_lq_paths.extend(sorted(["saltpepper" for x in Path(os.path.join(self.dataroot_gt, opt['saltpepper_path'])).rglob('*.' + "png")]))
                tmp_gt_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['saltpepper_path'])).rglob('*.' + "png")]))
                assert self.process_data_num <= len(tmp_lq_paths), f"The number of data should be less than the number of data in the folder={len(tmp_lq_paths)}"
         
                paired_paths = random.sample(list(zip(tmp_lq_paths, tmp_gt_paths)), self.process_data_num)
                tmp_lq_paths, tmp_gt_paths = zip(*paired_paths)
                self.lq_paths.extend(list(tmp_lq_paths))
                self.gt_paths.extend(list(tmp_gt_paths))
                tmp_lq_paths = []
                tmp_gt_paths = []
        if 'gaussian_path' in opt:
            if isinstance(opt['gaussian_path'], str):
                # Use rglob to recursively search for images
                tmp_lq_paths.extend(sorted(["gaussian" for x in Path(os.path.join(self.dataroot_gt, opt['gaussian_path'])).rglob('*.' + "png")]))
                tmp_gt_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['gaussian_path'])).rglob('*.' + "png")]))
                assert self.process_data_num <= len(tmp_lq_paths), f"The number of data should be less than the number of data in the folder={len(tmp_lq_paths)}"
                
                paired_paths = random.sample(list(zip(tmp_lq_paths, tmp_gt_paths)), self.process_data_num)
                tmp_lq_paths, tmp_gt_paths = zip(*paired_paths)
                self.lq_paths.extend(list(tmp_lq_paths))
                self.gt_paths.extend(list(tmp_gt_paths))
                tmp_lq_paths = []
                tmp_gt_paths = []
        if 'super_resolution_path' in opt:
            if isinstance(opt['super_resolution_path'], str):
                # Use rglob to recursively search for images
                tmp_lq_paths.extend(sorted(["low resolution" for x in Path(os.path.join(self.dataroot_gt, opt['super_resolution_path'])).rglob('*.' + "png")]))
                tmp_gt_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['super_resolution_path'])).rglob('*.' + "png")]))
                assert self.process_data_num <= len(tmp_lq_paths), f"The number of data should be less than the number of data in the folder={len(tmp_lq_paths)}"
                
                paired_paths = random.sample(list(zip(tmp_lq_paths, tmp_gt_paths)), self.process_data_num)
                tmp_lq_paths, tmp_gt_paths = zip(*paired_paths)
                self.lq_paths.extend(list(tmp_lq_paths))
                self.gt_paths.extend(list(tmp_gt_paths))
                self.gt_degradation_types.extend(["low resolution"] * len(tmp_lq_paths))
                tmp_lq_paths = []
                tmp_gt_paths = []
        if 'inpainting_path' in opt:
            if isinstance(opt['inpainting_path'], str):
                # Use rglob to recursively search for images
                tmp_lq_paths.extend(sorted(["inpainting" for x in Path(os.path.join(self.dataroot_gt, opt['inpainting_path'])).rglob('*.' + "png")]))
                tmp_gt_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['inpainting_path'])).rglob('*.' + "png")]))
                assert self.process_data_num <= len(tmp_lq_paths), f"The number of data should be less than the number of data in the folder={len(tmp_lq_paths)}"
                
                paired_paths = random.sample(list(zip(tmp_lq_paths, tmp_gt_paths)), self.process_data_num)
                tmp_lq_paths, tmp_gt_paths = zip(*paired_paths)
                self.lq_paths.extend(list(tmp_lq_paths))
                self.gt_paths.extend(list(tmp_gt_paths))
                tmp_lq_paths = []
                tmp_gt_paths = []
        if 'rainy_path' in opt:
            if isinstance(opt['rainy_path'], str) and opt['rainy_path'] != ' ':
                # Use rglob to recursively search for images
                tmp_lq_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['rainy_path'], 'input')).rglob('*.' + "jpg")]))
                tmp_gt_paths.extend(sorted([str(x) for x in Path(os.path.join(self.dataroot_gt, opt['rainy_path'], 'target')).rglob('*.' + "jpg")]))
                assert self.process_data_num <= len(tmp_lq_paths), f"The number of data should be less than the number of data in the folder={len(tmp_lq_paths)}"
          
                paired_paths = random.sample(list(zip(tmp_lq_paths, tmp_gt_paths)), self.process_data_num)
                tmp_lq_paths, tmp_gt_paths = zip(*paired_paths)
                self.lq_paths.extend(list(tmp_lq_paths))
                self.gt_paths.extend(list(tmp_gt_paths))
                self.gt_degradation_types.extend(["rain"] * len(tmp_lq_paths))
                tmp_lq_paths = []
                tmp_gt_paths = []
        print("Number of low-quality paths:", len(self.lq_paths))
        print("Number of ground truth paths:", len(self.gt_paths))


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load lq&gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        lq_path = self.lq_paths[index]
        degradation_type = self.gt_degradation_types[index]

        if lq_path == "jpeg compression artifact" or lq_path == "ringing" or lq_path == 'poisson' or lq_path == "saltpepper" or lq_path == "gaussian" or lq_path == "inpainting" or lq_path == "low resolution":
            gt_path = self.gt_paths[index]
            # print("*** current gt_path: ", gt_path)
            # avoid errors caused by high latency in reading files
            retry = 5
            while retry > 0:
                try:
                    gt_img_bytes = self.file_client.get(gt_path, 'gt')
                    image = Image.open(BytesIO(gt_img_bytes))
                    image.verify()  # 验证图像完整性
                    # print("Image data is complete and valid.")
                except (IOError, OSError) as e:
                    logger = get_root_logger()
                    logger.warning(f'File client error: {e}, remaining retry times: {retry - 1}')
                    print(f'File client error: {e}, remaining retry times: {retry - 1}')
                    # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                    # change another file to read
                    index = random.randint(0, self.__len__()-1)
                    gt_path = self.gt_paths[index]
                    time.sleep(1)  # sleep 1s for occasional server congestion
                else:
                    # print("**********Success in reading gt image type2222**********")
                    break
                finally:
                    # print(f'File client error, remaining retry times: {retry}')
                    retry -= 1
        else:
            gt_path = self.gt_paths[index]
            # print("*** current gt_path: ", gt_path)
            # avoid errors caused by high latency in reading files
            retry = 5
            while retry > 0:
                try:
                    lq_img_bytes = self.file_client.get(lq_path, 'lq')
                    image = Image.open(BytesIO(lq_img_bytes))
                    image.verify()  # 验证图像完整性
                    # print("Image data is complete and valid.")
                    gt_img_bytes = self.file_client.get(gt_path, 'gt')
                    image = Image.open(BytesIO(gt_img_bytes))
                    image.verify()  # 验证图像完整性
                    # print("Image data is complete and valid.")
                except (IOError, OSError) as e:
                    logger = get_root_logger()
                    logger.warning(f'File client error: {e}, remaining retry times: {retry - 1}')
                    print(f'File client error: {e}, remaining retry times: {retry - 1}')
                    # change another file to read
                    index = random.randint(0, self.__len__()-1)
                    lq_path = self.lq_paths[index]
                    gt_path = self.gt_paths[index]
                    time.sleep(1)  # sleep 1s for occasional server congestion
                else:
                    # print("**********Success in reading gt image type1111**********")
                    break
                finally:
                    # print(f'File client error, remaining retry times: {retry}')
                    retry -= 1

        gt_img = imfrombytes(gt_img_bytes, float32=True)
        lq_img = None

        sythetic_degradation = ["jpeg compression artifact", "ringing", "poisson", "saltpepper", "low resolution", "inpainting"]
        if any(x in lq_path for x in sythetic_degradation):
            lq_img = gt_img
        else:        
            lq_img = imfrombytes(lq_img_bytes, float32=True)

        # filter the dataset and remove images with too low quality
        # print("lq_img.shape: ", lq_img.shape)
        # print("lq_path: ", lq_path)
        # print("gt_img.shape: ", gt_img.shape)
        # print("gt_path: ", gt_path)
        assert lq_img.shape == gt_img.shape, "lq image should be the same size as gt image!"
        gt_img_size = os.path.getsize(gt_path)
        gt_img_size = gt_img_size / 1024

        
        # while gt_img.shape[0] * gt_img.shape[1] < 200*200 or gt_img_size<20:
        #     index = random.randint(0, self.__len__()-1)
        #     lq_path = self.lq_paths[index]
        #     gt_path = self.gt_paths[index]
    
        #     time.sleep(0.1)  # sleep 1s for occasional server congestion
        #     lq_img_bytes = self.file_client.get(lq_path)
        #     gt_img_bytes = self.file_client.get(gt_path)
        #     lq_img = imfrombytes(lq_img_bytes, float32=True)
        #     gt_img = imfrombytes(gt_img_bytes, float32=True)
        #     gt_img_size = os.path.getsize(gt_path)
        #     gt_img_size = gt_img_size / 1024
    
        # # -------------------- Do augmentation for training: flip, rotation -------------------- #
        # use_hflip = self.opt['use_hflip'] and random.random() < 0.5
        # use_vflip = self.opt['use_rot'] and random.random() < 0.5
        # use_rot = self.opt['use_rot'] and random.random() < 0.5
        # lq_img = augment(lq_img, use_hflip, use_vflip, use_rot)
        # gt_img = augment(gt_img, use_hflip, use_vflip, use_rot)
    
        # # crop or pad to 10
        # # TODO: 10 is hard-coded. You may change it accordingly
        # h, w = gt_img.shape[0:2]
        # crop_pad_size = self.crop_size
        # # pad
        # if h < crop_pad_size or w < crop_pad_size:
        #     pad_h = max(0, crop_pad_size - h)
        #     pad_w = max(0, crop_pad_size - w)
        #     lq_img = cv2.copyMakeBorder(lq_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        #     gt_img = cv2.copyMakeBorder(gt_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # # crop
        # if gt_img.shape[0] > crop_pad_size or gt_img.shape[1] > crop_pad_size:
        #     h, w = gt_img.shape[0:2]
        #     # randomly choose top and left coordinates
        #     top = random.randint(0, h - crop_pad_size)
        #     left = random.randint(0, w - crop_pad_size)
        #     # top = (h - crop_pad_size) // 2 -1
        #     # left = (w - crop_pad_size) // 2 -1
        #     lq_img = lq_img[top:top + crop_pad_size, left:left + crop_pad_size, ...]
        #     gt_img = gt_img[top:top + crop_pad_size, left:left + crop_pad_size, ...]
    
        # BGR to RGB, HWC to CHW, numpy to tensor
        lq_img = img2tensor(lq_img, bgr2rgb=True, float32=True)
        gt_img = img2tensor(gt_img, bgr2rgb=True, float32=True)

        preproccesor = ImageDegradation(configs=self.opt)
        if "jpeg compression artifact" in lq_path:
            lq_img = preproccesor.apply_jpeg_compression(lq_img)
            lq_img = torch.clamp(lq_img, 0, 1.0)
            lq_img = lq_img.contiguous()

            output_pil = transforms.ToPILImage()(lq_img.squeeze(0))
            outf = os.path.join("/data/zkl/AgenticIR/dataset/synthesis/LSDIR", f"{lq_path}/{index}.png")
            output_pil.save(outf)
            lq_path = outf
        elif "ringing" in lq_path:
            lq_img = preproccesor.apply_ringing_effect(lq_img)
        elif "poisson" in lq_path:
            lq_img = preproccesor.add_poisson_noise(lq_img)
        elif "saltpepper" in lq_path:
            lq_img = preproccesor.add_salt_and_pepper_noise(lq_img)
        elif "low resolution" in lq_path:
            lq_img = preproccesor.apply_super_resolution(lq_img)
            lq_img = torch.clamp(lq_img, 0, 1.0)
            lq_img = lq_img.contiguous()

            output_pil = transforms.ToPILImage()(lq_img.squeeze(0))
            outf = os.path.join("/data/zkl/AgenticIR/dataset/synthesis/LSDIR", f"{lq_path}/{index}.png")
            output_pil.save(outf)
            lq_path = outf
        elif "inpainting" in lq_path:
            lq_img = preproccesor.apply_image_inpainting(lq_img)

        # print("lq_img_shape: ", lq_img.shape)
        # print("gt_img_shape: ", gt_img.shape)
        # lq_output_pil = transforms.ToPILImage()(lq_img)
        # # lq_output_pil.save(f"lq.png")
        # gt_output_pil = transforms.ToPILImage()(gt_img)
        # # gt_output_pil.save(f"gt.png")
        lq_img = torch.clamp(lq_img, 0, 1.0)
        gt_img = torch.clamp(gt_img, 0, 1.0)

        lq_img = lq_img.contiguous()
        gt_img = gt_img.contiguous()


        # lq_img = lq_img * 2 - 1.0 # TODO 0~1?
        # gt_img = gt_img * 2 - 1.0

        # lq_img = torch.clamp(lq_img, -1.0, 1.0)
        # gt_img = torch.clamp(gt_img, -1.0, 1.0)
    
        return_dict = {'lq': lq_img, 'lq_path': lq_path, 'gt': gt_img, 'gt_path': gt_path, 'degradation_type': degradation_type}
        return return_dict
    
    def __len__(self):
        return len(self.lq_paths)


class ImageDegradation:
    def __init__(self, device='cpu', configs=None):
        """
        Initialize the ImageDegradation class.

        Args:
            device (str): Device to run the operations on ('cuda' or 'cpu').
            jpeg_quality_range (tuple): Range of JPEG quality values (min, max).
        """
        self.device = device
        self.configs = configs
        self.jpeger = DiffJPEG(differentiable=False).to(self.device)  # Simulate JPEG compression artifacts

    def apply_jpeg_compression(self, image):
        """
        Apply JPEG compression to the input image.

        Args:
            image (torch.Tensor): Input image tensor with values in [0, 1].

        Returns:
            torch.Tensor: JPEG-compressed image tensor.
        """
        # Ensure the input image is in the range [0, 1]
        image = image.to(memory_format=torch.contiguous_format).float()
        image = torch.clamp(image, 0, 1).unsqueeze(0)

        # Randomly sample a JPEG quality value within the specified range
        jpeg_p = image.new_zeros(image.size(0)).uniform_(*self.configs['jpeg_range_list'])
        # print(jpeg_p)

        # Apply JPEG compression
        compressed_image = self.jpeger(image, quality=jpeg_p)
        # print("compressed_image_shape: ", compressed_image.shape)

        return compressed_image.detach()
    
    def apply_ringing_effect(self, image):
        """
        Apply a synthetic ringing effect to the input image.

        Args:
            image (torch.Tensor): Input image tensor with values in [0, 1].

        Returns:
            torch.Tensor: Image with synthetic ringing artifacts.
        """
        image = image.unsqueeze(0)
        # Step 1: Apply Gaussian blur to smooth the image
        blurred_image = gaussian_blur(image, kernel_size=[5, 5], sigma=1.0)

        # Step 2: Sharpen the blurred image to enhance edges
        ringing_strength = random.uniform(*self.configs['ringing_strength_range'])
        sharpened_image = image + ringing_strength * (image - blurred_image)

        # Step 3: Clamp the values to [0, 1] to avoid out-of-range artifacts
        sharpened_image = torch.clamp(sharpened_image, 0, 1)

        # Step 4: Add high-frequency components in the frequency domain
        fft_image = torch.fft.fft2(image, dim=(-2, -1))
        fft_shifted = torch.fft.fftshift(fft_image, dim=(-2, -1))

        # Create a mask to amplify high-frequency components
        _, _, h, w = fft_shifted.shape
        mask = torch.ones_like(fft_shifted)
        center_h, center_w = h // 2, w // 2
        mask[:, :, center_h - 10:center_h + 10, center_w - 10:center_w + 10] = 0  # Suppress low frequencies
        amplified_fft = fft_shifted * mask * 1.5  # Amplify high frequencies

        # Inverse FFT to get the modified image
        ifft_shifted = torch.fft.ifftshift(amplified_fft, dim=(-2, -1))
        ringing_image = torch.fft.ifft2(ifft_shifted, dim=(-2, -1)).real

        # Combine the sharpened image and the ringing effect
        ringing_image = torch.clamp(sharpened_image + ringing_image, 0, 1)

        return ringing_image.squeeze(0)
    
    def add_poisson_noise(self, image):
        """
        Add Poisson noise to the input image.

        Args:
            image (torch.Tensor): Input image tensor with values in [0, 1].

        Returns:
            torch.Tensor: Image with Poisson noise added.
        """
        image = image.unsqueeze(0)
        gray_noise_prob = self.configs['gray_noise_prob']
        poisson_noisy_image = random_add_poisson_noise_pt(
            image,
            scale_range=self.configs['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)

        print("poisson_noisy_image_shape: ", poisson_noisy_image.shape)
        return poisson_noisy_image.squeeze(0)

    def add_salt_and_pepper_noise(self, image):
        """
        Add Salt-and-Pepper noise to the input image.

        Args:
            image (torch.Tensor): Input image tensor with values in [0, 1].

        Returns:
            torch.Tensor: Image with Salt-and-Pepper noise added.
        """
        image = image.unsqueeze(0)
        pepper_noisy_image = random_add_saltpepper_noise_pt(
            image,
            saltpepper_amount=self.configs['saltpepper_amount_range'],
            saltpepper_svsp=self.configs['saltpepper_svsp_range'],
        )

        return pepper_noisy_image.squeeze(0)
    
    def add_gaussian_noise(self, image):
        """
        Add Gaussian noise to the input image.

        Args:
            image (torch.Tensor): Input image tensor with values in [0, 1].

        Returns:
            torch.Tensor: Image with Gaussian noise added.
        """
        noisy_image = random_add_gaussian_noise_pt(
            image,
            sigma_range=self.configs['gaussian_noise_range'],
            gray_prob=self.configs['gaussian_noise_prob'],
        )

        return noisy_image
    
    def apply_super_resolution(self, image):

        """Degradation pipeline, modified from Real-ESRGAN:
        https://github.com/xinntao/Real-ESRGAN
        """
        image = image.unsqueeze(0)
        kernel_range = [2 * v + 1 for v in range(3, 11)]
        pulse_tensor = torch.zeros(21, 21).float()
        pulse_tensor[10, 10] = 1

        kernel_size = random.choice(kernel_range)
        if np.random.uniform() < self.configs['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.configs["kernel_list"],
                self.configs["kernel_prob"],
                kernel_size,
                self.configs["blur_sigma"],
                self.configs["blur_sigma"], [-math.pi, math.pi],
                self.configs["betag_range"],
                self.configs["betap_range"],
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(kernel_range)
        if np.random.uniform() < self.configs['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.configs["kernel_list2"],
                self.configs["kernel_prob2"],
                kernel_size,
                self.configs["blur_sigma2"],
                self.configs["blur_sigma2"], [-math.pi, math.pi],
                self.configs["betag_range2"],
                self.configs["betap_range2"],
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.configs['final_sinc_prob']:
            kernel_size = random.choice(kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)
        
        im_gt = image
        im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
        kernel1 = kernel
        sinc_kernel = sinc_kernel

        ori_h, ori_w = im_gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(im_gt, kernel1)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.configs['resize_prob'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.configs['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.configs['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.configs['gray_noise_prob']
        if random.random() < self.configs['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.configs['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.configs['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if random.random() < self.configs['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.configs['resize_prob2'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.configs['resize_range2'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.configs['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(int(ori_h / 4 * scale),
                    int(ori_w / 4 * scale)),
                mode=mode,
                )
        # add noise
        gray_noise_prob = self.configs['gray_noise_prob2']
        if random.random() < self.configs['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.configs['noise_range2'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.configs['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
                )

        if random.random() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // 4,
                        ori_w // 4),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // 4,
                        ori_w // 4),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)

        out = torch.clamp(out, 0, 1.0)

        lr_image = F.interpolate(
                out,
                size=(image.size(-2),
                      image.size(-1)),
                mode='bicubic',
                )
        # # clamp and round
        lr_image = torch.clamp(out, 0, 1.0)

        return lr_image.detach()
    
    def apply_image_inpainting(self, image):
        """
        对输入图像进行修复。
        Args:
            image (torch.Tensor): 输入图像张量，形状为 (C, H, W)，值范围为 [0, 1]。
        Returns:
            torch.Tensor: 修复后的图像张量，形状为 (C, H, W)。
        """
        # 假设 image 是一个 PyTorch 张量，形状为 (3, height, width)
        _, height, width = image.shape
        total_area = height * width  # 图像总面积
        target_area = 0.1 * total_area  # 目标面积

        # 创建一个全黑的掩码图像
        mask = np.zeros((height, width), dtype=np.uint8)

        def draw_lines():
            """绘制多条直线并返回其覆盖的总面积"""
            num_lines = random.randint(1, 5)
            total_line_area = 0
            for _ in range(num_lines):
                x1, y1 = random.randint(0, width - 1), random.randint(0, height - 1)
                x2, y2 = random.randint(0, width - 1), random.randint(0, height - 1)
                thickness = random.randint(5, 20)
                cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=thickness)
                # 计算直线覆盖的近似面积（矩形包围盒）
                total_line_area += abs(x2 - x1) * thickness + abs(y2 - y1) * thickness
            return total_line_area

        def draw_rectangle():
            """绘制矩形并返回其覆盖的面积"""
            min_side = int(np.sqrt(target_area))  # 确保矩形面积至少为目标面积
            x1, y1 = random.randint(0, width - min_side), random.randint(0, height - min_side)
            x2, y2 = random.randint(x1 + min_side, width), random.randint(y1 + min_side, height)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
            rect_area = (x2 - x1) * (y2 - y1)
            return rect_area

        def draw_circle():
            """绘制圆圈并返回其覆盖的面积"""
            min_radius = int(np.sqrt(target_area / np.pi))  # 确保圆圈面积至少为目标面积
            center_x = random.randint(min_radius, width - min_radius)
            center_y = random.randint(min_radius, height - min_radius)
            radius = random.randint(min_radius, min(center_x, center_y, width - center_x, height - center_y))
            cv2.circle(mask, (center_x, center_y), radius, 255, thickness=-1)
            circle_area = np.pi * radius ** 2
            return circle_area

        # 主循环：尝试生成形状，直到满足条件
        while True:
            shape_type = random.choice(["line", "rectangle", "circle"])
            if shape_type == "line":
                if draw_lines() >= target_area:
                    break
            elif shape_type == "rectangle":
                if draw_rectangle() >= target_area:
                    break
            elif shape_type == "circle":
                if draw_circle() >= target_area:
                    break
            # 如果未满足条件，清空掩码重新生成
            mask.fill(0)

        # 将掩码转换为 PyTorch 张量，并归一化到 [0, 1]
        mask_tensor = torch.from_numpy(mask).float() / 255.0  # 归一化到 [0, 1]
        mask_tensor = mask_tensor.unsqueeze(0)  # 添加通道维度 (1, H, W)
        mask_tensor = mask_tensor.expand(3, -1, -1)  # 扩展到三通道 (3, H, W)

        # 应用掩码到图像上
        masked_image = image * (1 - mask_tensor)
        masked_image = masked_image.unsqueeze_(0)

        return masked_image.squeeze(0)
