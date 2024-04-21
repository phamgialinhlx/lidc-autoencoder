import numpy as np
import glob
import os
from tqdm import tqdm  # Import tqdm

def add_segmentation(paths, dir):
    segmentation_path = os.path.join(dir, 'Image_Segmentation')
    _paths = []
    for img, mask in paths:
        file = '/'.join(img.split('/')[-2:])
        _paths.append((img, mask, os.path.join(segmentation_path, file)))
    return _paths

def get_paths(dir):
    image_path = os.path.join(dir, 'Image')
    mask_path = os.path.join(dir, 'Mask')
    image_files = glob.glob(os.path.join(image_path, '**/*.npy'), recursive=True)
    mask_files = glob.glob(os.path.join(mask_path, '**/*.npy'), recursive=True)
    image_files.sort()
    mask_files.sort()
    paths = []
    for img, mask in zip(image_files, mask_files):
        paths.append((img, mask))
    return paths

ROOT_DIR = '/work/hpc/pgl/LIDC-IDRI-Preprocessing/data/'
paths = []
paths += get_paths(ROOT_DIR) 
paths += get_paths(os.path.join(ROOT_DIR, 'Clean')) 
paths = add_segmentation(paths, ROOT_DIR)

TARGET_DIR = '/data/hpc/pgl/LIDC-IDRI-2D/data/'
CLEAN_DIR = os.path.join(TARGET_DIR, 'clean')
NODULE_DIR = os.path.join(TARGET_DIR, 'nodule')
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(NODULE_DIR, exist_ok=True)
os.makedirs(os.path.join(CLEAN_DIR, 'img'), exist_ok=True)
os.makedirs(os.path.join(CLEAN_DIR, 'mask'), exist_ok=True)
os.makedirs(os.path.join(CLEAN_DIR, 'segmentation'), exist_ok=True)
os.makedirs(os.path.join(NODULE_DIR, 'img'), exist_ok=True)
os.makedirs(os.path.join(NODULE_DIR, 'mask'), exist_ok=True)
os.makedirs(os.path.join(NODULE_DIR, 'segmentation'), exist_ok=True)

for image_file, mask_file, seg_file in tqdm(paths, desc='Processing images'):
    FILE_NAME = image_file.split('/')[-2]
    img = np.load(image_file)
    mask = np.load(mask_file)
    seg = np.load(seg_file)
    for i in range(img.shape[0]):
        img_i = img[i]
        mask_i = mask[i]
        seg_i = seg[i]
        if mask_i.sum() == 0:
            np.save(os.path.join(CLEAN_DIR, 'img', f'{FILE_NAME}_{i}.npy'), img_i)
            np.save(os.path.join(CLEAN_DIR, 'mask', f'{FILE_NAME}_{i}.npy'), mask_i)
            np.save(os.path.join(CLEAN_DIR, 'segmentation', f'{FILE_NAME}_{i}.npy'), seg_i)
        else:
            np.save(os.path.join(NODULE_DIR, 'img', f'{FILE_NAME}_{i}.npy'), img_i)
            np.save(os.path.join(NODULE_DIR, 'mask', f'{FILE_NAME}_{i}.npy'), mask_i)
            np.save(os.path.join(NODULE_DIR, 'segmentation', f'{FILE_NAME}_{i}.npy'), seg_i)

