
import os
import torch
import faiss
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from os.path import join
import torch.utils.data as data
import torchvision.transforms as T
from torch.utils.data.dataset import Subset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data.dataloader import DataLoader
import random
from torchvision.transforms import InterpolationMode
import torchvision.transforms as TVT
from viz_lidar import viz_lidar_open3d
import MinkowskiEngine as ME
import copy
import matplotlib.pyplot as plt
import utm
from pc_augmentation import (
    PCRandomFlip,
    PCRandomRotation,
    PCRandomTranslation,
    PCRandomScale,
    PCRandomShear,
    PCJitterPoints,
    PCRemoveRandomPoints,
    PCRemoveRandomBlock
)

from nuscenes.nuscenes import NuScenes

from nuscenes.utils.data_classes import LidarPointCloud

from tools.options import parse_arguments
opt = parse_arguments()







# trainselectlocationlist = [  # define which location to use
#     "2013_05_28_drive_0000_sync",
#     "2013_05_28_drive_0002_sync",
#     "2013_05_28_drive_0003_sync",
#     "2013_05_28_drive_0004_sync",
#     "2013_05_28_drive_0005_sync",
#     "2013_05_28_drive_0006_sync",
#     "2013_05_28_drive_0007_sync",
#     "2013_05_28_drive_0009_sync",
#     "2013_05_28_drive_0010_sync",
# ]

# testselectlocationlist = [
#     "2013_05_28_drive_0000_sync",
#     "2013_05_28_drive_0002_sync",
#     "2013_05_28_drive_0003_sync",
#     "2013_05_28_drive_0004_sync",
#     "2013_05_28_drive_0005_sync",
#     "2013_05_28_drive_0006_sync",
#     "2013_05_28_drive_0007_sync",
#     "2013_05_28_drive_0009_sync",
#     "2013_05_28_drive_0010_sync",
# ]


trainselectlocationlist = [
    'singapore-onenorth',
    'singapore-hollandvillage',
    'singapore-queenstown',
    'boston-seaport'
]

testselectlocationlist = [
    'singapore-onenorth',
    'singapore-hollandvillage',
    'singapore-queenstown',
    'boston-seaport'
]


train_ratio = opt.train_ratio
share_db = opt.share_db


base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def path_to_pil_img(path):
    image = Image.open(path)
    image = image.convert("RGB")
    return image




def nuscenes_collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (images,
        triplets_local_indexes, triplets_global_indexes).
        triplets_local_indexes are the indexes referring to each triplet within images.
        triplets_global_indexes are the global indexes of each image.
    Args:
        batch: list of tuple (images, triplets_local_indexes, triplets_global_indexes).
            considering each query to have 10 negatives (negs_num_per_query=10):
            - images: torch tensor of shape (12, 3, h, w).
            - triplets_local_indexes: torch tensor of shape (10, 3).
            - triplets_global_indexes: torch tensor of shape (12).
    Returns:
        images: torch tensor of shape (batch_size*12, 3, h, w).
        triplets_local_indexes: torch tensor of shape (batch_size*10, 3).
        triplets_global_indexes: torch tensor of shape (batch_size, 12).
    """
    # image
    # bev
    # pc
    # images = torch.cat([e[0]['image'] for e in batch]) # [12,c,h,w] -> [b*12,c,h,w]
    # bevs = torch.stack([e[0]['bev'] for e in batch]) # [3,h,w] -> [b,3,h,w]
    # sphs = torch.stack([e[0]['sph'] for e in batch]) # [3,h,w] -> [b,3,h,w]
    # pcs = [e[0]['pc'] for e in batch]
    query_pc = [e[0]['query_pc'] for e in batch]
    query_image = torch.stack([e[0]['query_image'] for e in batch]) 
    query_bev = torch.stack([e[0]['query_bev'] for e in batch])
    query_sph = torch.stack([e[0]['query_sph'] for e in batch])
    query_eastnorth = torch.stack([e[0]['query_eastnorth'] for e in batch])
    db_map = torch.stack([e[0]['db_map'] for e in batch])
    db_eastnorth = torch.stack([e[0]['db_eastnorth'] for e in batch])
    # positive_db_map = torch.stack([e[0]['positive_db_map'] for e in batch]) 
    # negative_db_maps = torch.stack([e[0]['negative_db_maps'] for e in batch]) 
    # ---- batch augmentation (quantize is in __getitem__)
    coords = ME.utils.batched_coordinates(query_pc)
    batchids = coords[:,:1]
    coords = coords[:,1:]
    coords = PCRandomRotation(max_theta=5, max_theta2=0, axis=np.array([0, 0, 1]))(coords) # CPU intense
    coords = torch.cat([batchids, coords], dim=1)
    feats = torch.ones([coords.shape[0], 1]).float()

    # feats = torch.ones([coords.shape[0], 1]).float()
    triplets_local_indexes = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        local_indexes += len(global_indexes) * i  # Increment local indexes by offset (len(global_indexes) is 12)
    output_dict = {
        # 'image': images,
        # 'bev': bevs,
        # 'sph': sphs,
        'coords': coords,
        'features': feats,
        'query_image': query_image,
        'query_bev': query_bev,
        'query_sph': query_sph,
        'query_eastnorth': query_eastnorth,
        # 'positive_db_map': positive_db_map,
        # 'negative_db_maps': negative_db_maps,
        'db_map': db_map,
        'db_eastnorth': db_eastnorth,
    }
    # return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes
    return output_dict, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes





def nuscenes_collate_fn_cache_db(batch):
    # images = torch.stack([e[0]['image'] for e in batch]) # [12,c,h,w] -> [b*12,c,h,w]
    # bevs = torch.empty([images.shape[0], 0])
    # sphs = torch.empty([images.shape[0], 0])
    # coords = torch.empty([images.shape[0], 0])
    # feats = torch.empty([images.shape[0], 0])
    # query_image = torch.stack([e[0]['query_image'] for e in batch])
    # positive_db_map = torch.stack([e[0]['positive_db_map'] for e in batch])
    # negative_db_maps = torch.stack([e[0]['negative_db_maps'] for e in batch])
    # query_image = torch.stack([e[0]['query_image'] for e in batch])
    # query_bev = torch.stack([e[0]['query_bev'] for e in batch])
    # query_sph = torch.stack([e[0]['query_sph'] for e in batch])
    db_map = torch.stack([e[0]['db_map'] for e in batch])
    indices = torch.tensor([e[1] for e in batch])
    db_location = [e[0]['db_location'] for e in batch]
    db_eastnorth = torch.stack([e[0]['db_eastnorth'] for e in batch])

    output_dict = {
        # 'image': images,
        # 'bev': bevs,
        # 'sph': sphs,
        # 'coord': coords,
        # 'feat': feats,
        # 'query_image': query_image,
        # 'positive_db_map': positive_db_map,
        # 'negative_db_maps': negative_db_maps,
        # 'query_image': query_image,
        # 'query_bev': query_bev,
        # 'query_sph': query_sph,
        'db_map': db_map,
        'db_location': db_location,
        'db_eastnorth': db_eastnorth,
    }
    return output_dict, indices




def nuscenes_collate_fn_cache_q(batch):
    '''
    output of collate_fn should be applicable with .to(device)
    '''
    # images = torch.stack([e[0]['image'] for e in batch]) # [12,c,h,w] -> [b*12,c,h,w]
    # bevs = torch.stack([e[0]['bev'] for e in batch]) # [3,h,w] -> [b,3,h,w]
    # # sphs = torch.stack([e[0]['sph'] for e in batch]) # [3,h,w] -> [b,3,h,w]
    # pcs = [e[0]['pc'] for e in batch]
    query_image = torch.stack([e[0]['query_image'] for e in batch])
    # query_image = query_image * 0
    query_bev = torch.stack([e[0]['query_bev'] for e in batch])
    query_sph = torch.stack([e[0]['query_sph'] for e in batch])
    query_pc = [e[0]['query_pc'] for e in batch]
    # ---- batch augmentation (quantize is in __getitem__)
    coords = ME.utils.batched_coordinates(query_pc)
    batchids = coords[:,:1]
    coords = coords[:,1:]
    coords = PCRandomRotation(max_theta=5, max_theta2=0, axis=np.array([0, 0, 1]))(coords) # CPU intense
    # coords = coords * 0 # for test
    coords = torch.cat([batchids, coords], dim=1)
    feats = torch.ones([coords.shape[0], 1]).float()
    query_location = [e[0]['query_location'] for e in batch]
    query_eastnorth = torch.stack([e[0]['query_eastnorth'] for e in batch])

    db_map = torch.stack([e[0]['db_map'] for e in batch])
    # positive_db_map = torch.stack([e[0]['positive_db_map'] for e in batch])
    # negative_db_maps = torch.stack([e[0]['negative_db_maps'] for e in batch])

    indices = torch.tensor([e[1] for e in batch])
    output_dict = {
        # 'image': images,
        # 'bev': bevs,
        # 'sph': sphs,
        'coords': coords,
        'features': feats,
        'query_image': query_image,
        'query_bev': query_bev,
        'query_sph': query_sph,
        # 'positive_db_map': positive_db_map,
        # 'negative_db_maps': negative_db_maps,
        'db_map': db_map,
        'query_location': query_location,
        'query_eastnorth': query_eastnorth,
    }
    return output_dict, indices







# def load_qimage(datapath, split):
#     image = Image.open(datapath)
#     image = image.convert('RGB')
#     if split == 'train':
#         tf = TVT.Compose([TVT.Resize(opt.q_resize), 
#                         # TVT.ColorJitter(brightness=opt.q_jitter, contrast=opt.q_jitter, saturation=opt.q_jitter, hue=min(0.5, opt.q_jitter)),
#                         TVT.ToTensor(),
#                           TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                         # TVT.Normalize(mean=0.5, std=0.22)
#                         ])
#     elif split == 'test':
#         tf = TVT.Compose([TVT.Resize(opt.q_resize), 
#                         TVT.ToTensor(),
#                           TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                         # TVT.Normalize(mean=0.5, std=0.22)
#                         ])
#     image = tf(image)
#     return image



def load_dbimage(datapath, split):
    image = Image.open(datapath)
    image = image.convert('RGB')
    if split == 'train':
        tf = TVT.Compose([
            # TVT.CenterCrop(opt.db_cropsize),
            TVT.Resize(opt.db_resize, antialias=True), 
            # TVT.ColorJitter(brightness=opt.db_jitter, contrast=opt.db_jitter, saturation=opt.db_jitter, hue=min(0.5, opt.db_jitter)),
            TVT.ToTensor(),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # TVT.Normalize(mean=0.5, std=0.22)
                        ])
    elif split == 'test':
        tf = TVT.Compose([
            # TVT.CenterCrop(opt.db_cropsize),
            TVT.Resize(opt.db_resize, antialias=True), 
            TVT.ToTensor(),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # TVT.Normalize(mean=0.5, std=0.22)
                        ])
    else:
        raise NotImplementedError
    image = tf(image)
    return image





def generate_bev_from_pc(pc, w=200, max_thd=100):
    """
    w+1: bev width
    max_thd: max threshold of x,y,z
    """
    pc = copy.deepcopy(pc)
    assert pc.shape[1] == 3
    # remove pc outside max_thd
    pc = pc[np.max(np.abs(pc), axis=1) < max_thd]
    bin_max = np.max(pc, axis=0)
    bin_min = np.min(pc, axis=0)
    assert np.all(bin_max <= max_thd)
    assert np.all(bin_min >= -max_thd)
    pc = pc + max_thd
    pc = pc / (2 * max_thd) * w
    pc = pc.astype(np.int64)
    bin_max = np.max(pc, axis=0)
    # print(bin_max)
    bev = np.zeros([w+1, w+1], dtype=np.float32)
    assert np.all(bin_max <= bev.shape[0])
    bev[pc[:, 0], pc[:, 1]] = pc[:, 2]
    return bev




def generate_sph_from_pc(pc, w=361, h=61):
    # kitti   w=361  h=61
    # ithaca  w=361  h=101
    # kitti360 w=361 h=61
    # w = 361
    # h = 61
    # if 'ithaca365' in opt.dataset_name:
    #     w = 361
    #     h = 101

    # generate spherical projection from pc
    pc = copy.deepcopy(pc)
    assert pc.shape[1] == 3
    # u-v : h-w
    # u: 
    u = np.arctan2(pc[:,2], np.sqrt(pc[:,0]**2 + pc[:,1]**2))
    u = u / np.pi * 180
    u = u + 25 
    u = u * 2
    u = h - u
    # v: [0, 360]
    v = np.arctan2(pc[:,0], pc[:,1])
    v = v / np.pi * 180
    v = v + 180
    r = np.sqrt(pc[:,0]**2 + pc[:,1]**2 + pc[:,2]**2)
    uv = np.stack([u, v], axis=1)
    uv = np.array(uv, dtype=np.int32)
    # plt.scatter(uv[:,1], uv[:,0], s=1, c=r, cmap='jet')
    # plt.show()
    if 'ithaca365' in opt.dataset_name:
        ids_h = (uv[:,0] < h) &  (uv[:,0] >= 0)
        uv = uv[ids_h]
        r = r[ids_h]
    sph = np.zeros([h, w])
    sph[uv[:,0], uv[:,1]] = r
    # if 'ithaca365' in opt.dataset_name:
    #     sph = sph[25:80]
    # plt.imshow(sph)
    # plt.imsave('sph.png', sph)
    # plt.close()
    # plt.show()
    return sph
    



def load_pc_bev(file_path, split): # filename is the same as load_pc
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1,3) # for kitti360_voxel
    bev = generate_bev_from_pc(pc, w=200, max_thd=100)
    # ==== bev
    bev = Image.fromarray(bev).convert('RGB')
    # bev.show()
    (_w,_h) = bev.size
    assert opt.bev_resize <= 1
    resize_ratio = random.uniform(opt.bev_resize, 2-opt.bev_resize)
    resize_size = int(resize_ratio*min(_w,_h))
    assert resize_size >= opt.bev_cropsize

    if opt.bev_resize_mode == 'nearest':
        resize_mode = InterpolationMode.NEAREST
    elif opt.bev_resize_mode == 'bilinear':
        resize_mode = InterpolationMode.BILINEAR
    else:
        raise NotImplementedError
    
    if opt.bev_rotate_mode == 'nearest':
        rotate_mode = InterpolationMode.NEAREST
    elif opt.bev_rotate_mode == 'bilinear':
        rotate_mode = InterpolationMode.BILINEAR
    else:
        raise NotImplementedError
    
    if split == 'train':
        tf = TVT.Compose([
            TVT.Resize(resize_size, interpolation=resize_mode, antialias=True),
            TVT.RandomRotation(opt.bev_rotate, interpolation=rotate_mode),
            TVT.CenterCrop(opt.bev_cropsize),
            TVT.ColorJitter(brightness=opt.bev_jitter, contrast=opt.bev_jitter, saturation=opt.bev_jitter, hue=min(0.5, opt.bev_jitter)),
            TVT.ToTensor(),
            TVT.Normalize(mean=opt.bev_mean, std=opt.bev_std)
        ])
    elif split == 'test':
        tf = TVT.Compose([
            TVT.Resize(resize_size, interpolation=resize_mode, antialias=True),
            TVT.CenterCrop(opt.bev_cropsize),
            TVT.ToTensor(),
            TVT.Normalize(mean=opt.bev_mean, std=opt.bev_std)
        ])
    else:
        raise NotImplementedError
    bev = tf(bev)
    # # ---- DEBUG:viz
    # bev = bev*opt.bev_std + opt.bev_mean
    # bev = TVT.ToPILImage()(bev)
    # bev.show()
    return pc, bev



def load_pc_sph(file_path, split): # filename is the same as load_pc
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1,3) # for kitti360_voxel
    sph = generate_sph_from_pc(pc, w=361, h=61)
    # ==== sph
    sph = Image.fromarray(sph).convert('RGB')
    (_w,_h) = sph.size
    assert opt.sph_resize <= 1
    resize_ratio = random.uniform(opt.sph_resize, 2-opt.sph_resize)
    resize_size = int(resize_ratio*min(_w,_h))
    if split == 'train':
        tf = TVT.Compose([
            TVT.Resize(resize_size, interpolation=InterpolationMode.NEAREST, antialias=True),
            TVT.ColorJitter(brightness=opt.sph_jitter, contrast=opt.sph_jitter, saturation=opt.sph_jitter, hue=min(0.5, opt.sph_jitter)),
            TVT.ToTensor(),
            TVT.Normalize(mean=opt.sph_mean, std=opt.sph_std)
        ])
    elif split == 'test':
        tf = TVT.Compose([
            TVT.Resize(resize_size, interpolation=InterpolationMode.NEAREST, antialias=True),
            TVT.ToTensor(),
            TVT.Normalize(mean=opt.sph_mean, std=opt.sph_std)
        ])
    else:
        raise NotImplementedError
    sph = tf(sph)
    return pc, sph














def get_location_from_sample_token(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    scene_token = sample['scene_token']
    location = nusc.get('log', nusc.get('scene', scene_token)['log_token'])['location']
    return location






def get_latloneastnorth_from_sample_token(nusc, sample_token, location):
    sample = nusc.get('sample', sample_token)
    ego_pose = nusc.get('ego_pose', sample['data']['LIDAR_TOP'])
    ego_pose = ego_pose['translation']
    ego_pose = np.array(ego_pose)
    if location == 'boston-seaport':
        eastingorg, northingorg, zone_number, zone_letter = utm.from_latlon(42.336849169438615, -71.05785369873047)
        # clock-wise rotate
        degree = 1.5
        R = np.array([[np.cos(np.pi/180*degree), -np.sin(np.pi/180*degree)], 
                      [np.sin(np.pi/180*degree), np.cos(np.pi/180*degree)]])
        ego_pose[0:2] = ego_pose[0:2] @ R 
    elif location == 'singapore-onenorth':
        eastingorg, northingorg, zone_number, zone_letter = utm.from_latlon(1.2882100868743724, 103.78475189208984)
    elif location == 'singapore-hollandvillage':
        eastingorg, northingorg, zone_number, zone_letter = utm.from_latlon(1.2993652317780957, 103.78217697143555)
    elif location == 'singapore-queenstown':
        eastingorg, northingorg, zone_number, zone_letter = utm.from_latlon(1.2782562240223188, 103.76741409301758)
    else:
        raise NotImplementedError
    ego_pose[0] = ego_pose[0] + eastingorg
    ego_pose[1] = ego_pose[1] + northingorg
    easting = ego_pose[0]
    northing = ego_pose[1]
    latlon = utm.to_latlon(easting, northing, zone_number, zone_letter)
    outputdict = {
        'lat': latlon[0],
        'lon': latlon[1],
        'east': easting,
        'north': northing,
        'zone_number': zone_number,
        'zone_letter': zone_letter
    }
    return outputdict






def get_datapaths_from_sample_token(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    sensornames = ['LIDAR_TOP', 
                   'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    datapaths = {}
    for sensorname in sensornames:
        datatoken = sample['data'][sensorname]
        data = nusc.get('sample_data', datatoken)
        dataname = data['filename']
        datapath = os.path.join(nusc.dataroot, dataname)
        # assert os.path.exists(datapath), f'{datapath} does not exist'
        datapaths[sensorname] = datapath

    return datapaths







def load_sensordata_from_sampletoken(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    sensornames = ['LIDAR_TOP', 
                   'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    sensordatas = {}
    for sensorname in sensornames:
        datatoken = sample['data'][sensorname]
        data = nusc.get('sample_data', datatoken)
        dataname = data['filename']
        datapath = os.path.join(nusc.dataroot, dataname)
        if sensorname == 'LIDAR_TOP':
            # load pc using .pcd.bin
            # pc = LidarPointCloud.from_file(datapath).points.T # [n,4]
            # pc = pc[:,:3] # [n,3]
            # load pc using .npy
            pcpath = datapath.replace('.pcd.bin', '.npy')
            pcpath = pcpath.split('/')
            pcpath[-2] += f'_voxel{1}'
            pcpath = '/'.join(pcpath)
            pc = np.load(pcpath, allow_pickle=True)
            pc = ME.utils.sparse_quantize(coordinates=pc, quantization_size=opt.quant_size)
            sensordatas['LIDAR_TOP'] = pc
            
            # load sph
            if 'lcpr' in opt.modelq:
                sphdatapath = datapath.replace('.pcd.bin', '.npy')
                sphdatapath = sphdatapath.replace('samples/LIDAR_TOP', 'samples/RANGE_DATA')
                sph = np.load(sphdatapath)
                assert len(sph.shape) == 2
                tf = TVT.Compose([TVT.ToTensor(),
                                TVT.Resize(32, antialias=True), # [1,32,1056]
                                ])
                sph = tf(sph)
                if sph.shape[0] == 1:
                    sph = sph.repeat(3,1,1)
                sensordatas['RANGE_DATA'] = sph
            else:
                sensordatas['RANGE_DATA'] = torch.empty(0)

            # load bev
            # w = 200
            # max_thd = 100
            # bevdatapath = datapath.replace('.pcd.bin', '.png')
            # bevdatapath = bevdatapath.replace('samples/LIDAR_TOP', f'samples/BEV_DATA_w{w}_thd{max_thd}')
            # bev = Image.open(bevdatapath).convert('RGB')
            # tf = TVT.Compose([TVT.ToTensor(),
            #                 TVT.Resize(256, antialias=True), # [3,32,32]
            #                 ])
            # bev = tf(bev)
            # sensordatas['BEV_DATA'] = bev
            sensordatas['BEV_DATA'] = torch.empty(0)
        else: # for cameras
            datapath = datapath.split('/')
            datapath[-2] += f'_size{256}' 
            datapath = '/'.join(datapath)
            image = Image.open(datapath)
            tf = TVT.Compose([TVT.Resize(192, antialias=True),  # 192 to support 4090
                              TVT.ToTensor(),
                              TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                              ])
            image = tf(image)   
            sensordatas[sensorname] = image

    # Stack
    qcam = [] 
    camnames = opt.camnames.split('_')
    for camname in camnames:
        if camname == 'f':
            qcam.append(sensordatas['CAM_FRONT'])
        elif camname == 'fl':
            qcam.append(sensordatas['CAM_FRONT_LEFT'])
        elif camname == 'fr':
            qcam.append(sensordatas['CAM_FRONT_RIGHT'])
        elif camname == 'b':
            qcam.append(sensordatas['CAM_BACK'])
        elif camname == 'bl':
            qcam.append(sensordatas['CAM_BACK_LEFT'])
        elif camname == 'br':
            qcam.append(sensordatas['CAM_BACK_RIGHT'])
        else:
            raise NotImplementedError
    # qcam = torch.stack(qcam, dim=0) # [ncam,3,h,w]
    qcam = torch.cat(qcam, dim=-1) # [3,h,w*ncam] panorama cat
    # TODO: support multi-cam setting
    # assert qcam.shape[0] == 1
    # qcam = qcam[0]

    # qcam = torch.stack([sensordatas['CAM_FRONT'], 
    #                     sensordatas['CAM_FRONT_LEFT'], 
    #                     sensordatas['CAM_FRONT_RIGHT'], 
    #                     sensordatas['CAM_BACK'], 
    #                     sensordatas['CAM_BACK_LEFT'], 
    #                     sensordatas['CAM_BACK_RIGHT']], dim=0) # [ncam,3,h,w]

    return qcam, sensordatas['RANGE_DATA'], sensordatas['BEV_DATA'], sensordatas['LIDAR_TOP']



def get_seq_sample_tokens(nusc, sampletoken, seq_len, current_frame_type):
    # current_frame_type: ['new', 'mid', 'old']
    sample = nusc.get('sample', sampletoken)
    seq_sample_tokens = []
    if current_frame_type == 'new':
        for i in range(seq_len):
            seq_sample_tokens = [sample['token']] + seq_sample_tokens
            sampletoken = sample['prev']
            if len(sampletoken) == 0:
                sampletoken = sample['token']
            sample = nusc.get('sample', sampletoken)

    elif current_frame_type == 'old':
        for i in range(seq_len):
            seq_sample_tokens.append(sample['token'])
            sampletoken = sample['next']
            if len(sampletoken) == 0:
                sampletoken = sample['token']
            sample = nusc.get('sample', sampletoken)

    elif current_frame_type == 'mid':
        midsample = sample
        seq_sample_tokens.append(sample['token']) # middle one
        for i in range(seq_len//2):
            sampletoken = sample['prev']
            if len(sampletoken) == 0:
                sampletoken = sample['token']
            sample = nusc.get('sample', sampletoken)
            seq_sample_tokens = [sample['token']] + seq_sample_tokens
        sample = midsample
        for i in range(seq_len//2):
            sampletoken = sample['next']
            if len(sampletoken) == 0:
                sampletoken = sample['token']
            sample = nusc.get('sample', sampletoken)
            seq_sample_tokens.append(sample['token'])

    assert len(seq_sample_tokens) == seq_len
    return seq_sample_tokens










class PCADataset(data.Dataset):
    def __init__(self, args, datasets_folder="dataset", dataset_folder="pitts30k/images/train"):
        dataset_folder_full_path = join(datasets_folder, dataset_folder)
        if not os.path.exists(dataset_folder_full_path):
            raise FileNotFoundError(f"Folder {dataset_folder_full_path} does not exist")
        self.images_paths = sorted(glob(join(dataset_folder_full_path, "**", "*.jpg"), recursive=True))
    
    def __getitem__(self, index):
        data_dict = {
            'image': base_transform(path_to_pil_img(self.images_paths[index]))
        }
        # return base_transform(path_to_pil_img(self.images_paths[index]))
        return data_dict
    
    def __len__(self):
        return len(self.images_paths)









# =================== test/cache dataset

class NuScenesBaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train"):
        super().__init__()
        self.dataset_name = dataset_name
        
        self.resize = args.resize
        self.test_method = args.test_method
        self.split = split
        

        # ==========
        # Use  v1.0-trainval_singapore-onenorth  training
        dataroot = opt.dataroot
        if split == 'train':
            selectlocationlist = trainselectlocationlist
            selectversion = 'v1.0-trainval'
        elif split == 'test':
            selectlocationlist = testselectlocationlist
            selectversion = 'v1.0-test'
        else:
            raise NotImplementedError
        
        self.nusc = NuScenes(version=selectversion, 
                             dataroot=dataroot, 
                             verbose=False)

        self.queries_infos = []
        self.queries_utms = [] 
        self.location_count = {
            'singapore-onenorth': 0,
            'singapore-hollandvillage': 0,
            'singapore-queenstown': 0,
            'boston-seaport': 0,
        }
        for isample, sample in enumerate(self.nusc.sample):
            if split == 'train':
                if isample % opt.traindownsample!= 0: # using 1/2 samples for training
                    continue
            elif split == 'test': # using all samples for testing
                None
            else:
                raise NotImplementedError
            sampletoken = sample['token']
            location = get_location_from_sample_token(self.nusc, sampletoken)
            if location in selectlocationlist:
                latloneastnorth = get_latloneastnorth_from_sample_token(self.nusc, sampletoken, location)
                datapaths = get_datapaths_from_sample_token(self.nusc, sampletoken)
                qsampleinfo = {
                    'sampletoken': sampletoken,
                    'nextsampletoken': sample['next'],
                    'location': location,
                    'lat': latloneastnorth['lat'],
                    'lon': latloneastnorth['lon'],
                    'east': latloneastnorth['east'],
                    'north': latloneastnorth['north'],
                }
                qsampleinfo.update(datapaths) 
                self.queries_infos.append(qsampleinfo)
                self.queries_utms.append([latloneastnorth['east'], latloneastnorth['north']])
                self.location_count[location] += 1
        self.queries_utms = np.array(self.queries_utms, dtype=np.float32)
        for location in self.location_count.keys():
            print(f"Number of q samples in {selectversion}_{split}_{location}: {self.location_count[location]}")
        print(f"Number of q samples in {selectversion}_{split}_{selectlocationlist}: {len(self.queries_infos)}")
        
        

        # for selectlocation in selectlocationlist:
        #     qpcdir = os.path.join(dataroot, 'data_3d_voxel0.5', selectlocation, 'velodyne_points/data')
        #     qposedir = os.path.join(dataroot, 'data_poses', selectlocation, 'oxts/data')
        #     qimage02dir = os.path.join(dataroot, 'data_2d_raw', selectlocation, 'image_02/data_rgb')
        #     qimage03dir = os.path.join(dataroot, 'data_2d_raw', selectlocation, 'image_03/data_rgb')
        #     qimage0203dir = os.path.join(dataroot, 'data_2d_cat0203', selectlocation, 'image_0203/data_rgb')
        #     qimage0203names = sorted(os.listdir(qimage0203dir))
        #     if split == 'train':
        #         qimage0203names = qimage0203names[:int(len(qimage0203names)*train_ratio)]
        #     elif split == 'test':
        #         qimage0203names = qimage0203names[int(len(qimage0203names)*train_ratio):]
        #     print(f"Number of q samples in {selectlocation}: {len(qimage0203names)}")
        #     for i_sample, qimage0203name in enumerate(qimage0203names):
        #         if i_sample % opt.traindownsample != 0: # downsample to accelerate training
        #             continue
        #         qpcpath = os.path.join(qpcdir, qimage0203name.replace('.png','.bin'))
        #         qimage02path = os.path.join(qimage02dir, qimage0203name.replace('.png','.png'))
        #         qimage03path = os.path.join(qimage03dir, qimage0203name.replace('.png','.png'))
        #         qimage0203path = os.path.join(qimage0203dir, qimage0203name.replace('.png','.png'))
        #         qposepath = os.path.join(qposedir, qimage0203name.replace('.png','.txt'))
        #         if not os.path.exists(qpcpath): continue
        #         if not os.path.exists(qposepath): continue
        #         qpose = open(qposepath).readline().split(' ')
        #         lat, lon = float(qpose[0]), float(qpose[1])
        #         east, north, _, _ = utm.from_latlon(lat, lon)
        #         qsampleinfo = {
        #             'lat': lat,
        #             'lon': lon,
        #             'east': east,
        #             'north': north,
        #             'qposepath': qposepath,
        #             'qimage02path': qimage02path,
        #             'qimage03path': qimage03path,
        #             'qimage0203path': qimage0203path,
        #             'qpcpath': qpcpath,
        #             'location': selectlocation,
        #         }
        #         self.queries_infos.append(qsampleinfo)
        #         self.queries_utms.append([east, north])
        # self.queries_utms = np.array(self.queries_utms, dtype=np.float32)
        # print(f"Number of q samples in {split}: {len(self.queries_infos)}")



        # Merge all db
        self.database_utms = []
        self.database_infos = []
        scale = 1
        zoom = 20   # higher is closer
        size = 320
        # maptype = opt.maptype
        # maptype = 'satellite'
        # maptype = 'osm'

        # for selectlocation in selectlocationlist:
        #     dbdir = os.path.join(dataroot, f'aerial_{selectversion}_{selectlocation}_{scale}_{zoom}_{size}_{maptype}')
        #     dbnames = os.listdir(dbdir)
        #     dbnames = sorted(dbnames)
        #     dbeastnorths = [e.split('@')[1:3] for e in dbnames]
        #     self.dbeastnorths += dbeastnorths
        #     dbpaths = [os.path.join(dbdir, dbname) for dbname in dbnames]
        #     self.dbpaths += dbpaths
        #     print(f"Number of aerial db in {selectversion}_{selectlocation}: {len(dbpaths)}")
        # self.dbeastnorths = np.array(self.dbeastnorths, dtype=np.float32)


        for selectlocation in selectlocationlist:
            # db_satellite_dir = os.path.join(dataroot, f'data_aerial_{scale}_{zoom}_{size}_satellite', selectlocation)
            # db_roadmap_dir = os.path.join(dataroot, f'data_aerial_{scale}_{zoom}_{size}_roadmap', selectlocation)
            db_satellite_dir = os.path.join(dataroot, f'aerial_{selectversion}_{selectlocation}_{scale}_{zoom}_{size}_satellite')
            db_roadmap_dir = os.path.join(dataroot, f'aerial_{selectversion}_{selectlocation}_{scale}_{zoom}_{size}_roadmap')
            dbnames = os.listdir(db_satellite_dir)
            dbnames = sorted(dbnames)
            if share_db == True:
                dbnames = dbnames
            elif share_db == False:
                dbnames = dbnames
                # if split == 'train':
                #     dbnames = dbnames[:int(len(dbnames)*train_ratio)]
                # elif split == 'test':
                #     dbnames = dbnames[int(len(dbnames)*train_ratio):]
            for i_dbname, dbname in enumerate(dbnames):
                if split == 'train':
                    if i_dbname % opt.traindownsample != 0: 
                        continue
                elif split == 'test':
                    None
                dbname_pure = dbname.replace('.png','')
                dbeastnorth = dbname_pure.split('@')[1:3]
                dblatlon = dbname_pure.split('@')[3:5]
                east, north = float(dbeastnorth[0]), float(dbeastnorth[1])
                lat, lon = float(dblatlon[0]), float(dblatlon[1])
                db_satellite_path = os.path.join(db_satellite_dir, dbname)
                db_roadmap_path = os.path.join(db_roadmap_dir, dbname)
                dbsampleinfo = {
                    'lat': lat,
                    'lon': lon,
                    'east': east,
                    'north': north,
                    'db_satellite_path': db_satellite_path,
                    'db_roadmap_path': db_roadmap_path,
                    'location': selectlocation,
                }
                self.database_infos.append(dbsampleinfo)
                self.database_utms.append([east, north])
        self.database_utms = np.array(self.database_utms, dtype=np.float32)
        print(f"Number of aerial db in {split}: {len(self.database_infos)}")




        # Find positive and negative 
        knn = NearestNeighbors(n_jobs=opt.num_workers+1)
        knn.fit(self.database_utms)
        softposthd = opt.val_positive_dist_threshold  # 25
        self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                             radius=softposthd,
                                                             return_distance=False)
        
        self.database_queries_infos = self.database_infos + self.queries_infos  # db + q
        self.database_num = len(self.database_infos)
        self.queries_num = len(self.queries_infos)
        print(f"Number of db+q: {len(self.database_queries_infos)}")
        a=1

    
    def __getitem__(self, index):
        if index >= self.database_num: # query
            # query_image = load_qimage(datapath=self.database_queries_infos[index]['qimage0203path'],split=self.split) 
            # query_pc, query_bev = load_pc_bev(file_path=self.database_queries_infos[index]['qpcpath'],split=self.split)
            sampletoken = self.database_queries_infos[index]['sampletoken']
            query_image, query_sph, query_bev, query_pc = load_sensordata_from_sampletoken(self.nusc, sampletoken)
            query_location = self.database_queries_infos[index]['location']
            query_eastnorth = torch.tensor([self.database_queries_infos[index]['east'], self.database_queries_infos[index]['north']]).float()
            db_location = None
            db_eastnorth = torch.empty(0)
            # query_pc = torch.empty(0)
            db_map = torch.empty(0)
            
        else: # database
            query_image = torch.empty(0)
            query_pc, query_bev, query_sph = torch.empty(0), torch.empty(0), torch.empty(0)
            query_location = None
            query_eastnorth = torch.empty(0)
            maptype = opt.maptype.split('_')
            db_map = []
            for each_maptype in maptype:
                if each_maptype == 'satellite':
                    each_db_map = load_dbimage(datapath=self.database_queries_infos[index]['db_satellite_path'], split=self.split)
                elif each_maptype == 'roadmap':
                    each_db_map = load_dbimage(datapath=self.database_queries_infos[index]['db_roadmap_path'], split=self.split)
                db_map.append(each_db_map) 
            db_map = torch.stack(db_map, 0) # [nmap,3,h,w]
            db_location = self.database_queries_infos[index]['location']
            db_eastnorth = torch.tensor([self.database_queries_infos[index]['east'], self.database_queries_infos[index]['north']]).float()


        output_dict = {
            'query_image': query_image, 
            'query_bev': query_bev,
            'query_sph': query_sph,
            'query_pc': query_pc,
            'query_location': query_location,
            'query_eastnorth': query_eastnorth,
            'db_map': db_map,
            'db_location': db_location,
            'db_eastnorth': db_eastnorth,
        }
        return output_dict, index


    
    def _test_query_transform(self, img):
        """Transform query image according to self.test_method."""
        C, H, W = img.shape
        if self.test_method == "single_query":
            # self.test_method=="single_query" is used when queries have varying sizes, and can't be stacked in a batch.
            processed_img = T.functional.resize(img, min(self.resize), antialias=True)
        elif self.test_method == "central_crop":
            # Take the biggest central crop of size self.resize. Preserves ratio.
            scale = max(self.resize[0]/H, self.resize[1]/W)
            processed_img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            processed_img = T.functional.center_crop(processed_img, self.resize)
            assert processed_img.shape[1:] == torch.Size(self.resize), f"{processed_img.shape[1:]} {self.resize}"
        elif self.test_method == "five_crops" or self.test_method == 'nearest_crop' or self.test_method == 'maj_voting':
            # Get 5 square crops with size==shorter_side (usually 480). Preserves ratio and allows batches.
            shorter_side = min(self.resize)
            processed_img = T.functional.resize(img, shorter_side)
            processed_img = torch.stack(T.functional.five_crop(processed_img, shorter_side))
            assert processed_img.shape == torch.Size([5, 3, shorter_side, shorter_side]), \
                f"{processed_img.shape} {torch.Size([5, 3, shorter_side, shorter_side])}"
        return processed_img
    
    def __len__(self):
        return len(self.database_queries_infos)
    
    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >"
    
    def get_positives(self):
        return self.soft_positives_per_query















# =================== train dataset

class NuScenesTripletsDataset(NuScenesBaseDataset):
    """Dataset used for training, it is used to compute the triplets
    with TripletsDataset.compute_triplets() with various mining methods.
    If is_inference == True, uses methods of the parent class BaseDataset,
    this is used for example when computing the cache, because we compute features
    of each image, not triplets.
    """
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train", negs_num_per_query=10):
        super().__init__(args, datasets_folder, dataset_name, split)
        self.mining = args.mining
        self.neg_samples_num = args.neg_samples_num  # Number of negatives to randomly sample
        self.negs_num_per_query = negs_num_per_query  # Number of negatives per query in each batch
        if self.mining == "full":  # "Full database mining" keeps a cache with last used negatives
            self.neg_cache = [np.empty((0,), dtype=np.int32) for _ in range(self.queries_num)]
        self.is_inference = False
        self.split = split


        # Find hard_positives_per_query, which are within train_positives_dist_threshold (10 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        hardposthd = opt.train_positives_dist_threshold
        self.hard_positives_per_query = list(knn.radius_neighbors(self.queries_utms,
                                             radius=hardposthd,  # 10 meters
                                             return_distance=False))
        
        #### Some queries might have no positive, we should remove those queries.
        queries_without_any_hard_positive = np.where(np.array([len(p) for p in self.hard_positives_per_query], dtype=object) == 0)[0]
        if len(queries_without_any_hard_positive) != 0:
            logging.info(f"There are {len(queries_without_any_hard_positive)} queries without any positives " +
                         "within the training set. They won't be considered as they're useless for training.")
        # Remove queries without positives
        # self.hard_positives_per_query = np.delete(self.hard_positives_per_query, queries_without_any_hard_positive)
        self.hard_positives_per_query = [p for i, p in enumerate(self.hard_positives_per_query) if i not in queries_without_any_hard_positive]


        self.queries_infos = [e for i, e in enumerate(self.queries_infos) if i not in queries_without_any_hard_positive]
        self.database_queries_infos = self.database_infos + self.queries_infos  # db + q
        self.queries_num = len(self.queries_infos)


        # msls_weighted refers to the mining presented in MSLS paper's supplementary.
        # Basically, images from uncommon domains are sampled more often. Works only with MSLS dataset.
        if self.mining == "msls_weighted":
            notes = [p.split("@")[-2] for p in self.queries_paths]
            try:
                night_indexes = np.where(np.array([n.split("_")[0] == "night" for n in notes]))[0]
                sideways_indexes = np.where(np.array([n.split("_")[1] == "sideways" for n in notes]))[0]
            except IndexError:
                raise RuntimeError("You're using msls_weighted mining but this dataset " +
                                   "does not have night/sideways information. Are you using Mapillary SLS?")
            self.weights = np.ones(self.queries_num)
            assert len(night_indexes) != 0 and len(sideways_indexes) != 0, \
                "There should be night and sideways images for msls_weighted mining, but there are none. Are you using Mapillary SLS?"
            self.weights[night_indexes] += self.queries_num / len(night_indexes)
            self.weights[sideways_indexes] += self.queries_num / len(sideways_indexes)
            self.weights /= self.weights.sum()
            logging.info(f"#sideways_indexes [{len(sideways_indexes)}/{self.queries_num}]; " +
                         "#night_indexes; [{len(night_indexes)}/{self.queries_num}]")
    

    def __getitem__(self, index):
        if self.is_inference:
            # At inference time return the single image. This is used for caching or computing NetVLAD's clusters
            return super().__getitem__(index)

        query_index, best_positive_index, neg_indexes = torch.split(self.triplets_global_indexes[index], (1, 1, self.negs_num_per_query))
        
        # query_image = load_qimage(datapath=self.queries_infos[query_index]['qimage0203path'],split=self.split) # [3,h,w]
        # query_pc, query_bev = load_pc_bev(file_path=self.queries_infos[query_index]['qpcpath'],split=self.split)
        
        # sampletoken = self.database_queries_infos[query_index]['sampletoken']
        sampletoken = self.queries_infos[query_index]['sampletoken']
        query_image, query_sph, query_bev, query_pc = load_sensordata_from_sampletoken(self.nusc, sampletoken)
        # query_location = self.queries_infos[query_index]['location']
        query_eastnorth = torch.tensor([self.queries_infos[query_index]['east'], self.queries_infos[query_index]['north']]).float()
        # query_pc = torch.empty(0)


        # positive_db_satellite_map = load_dbimage(datapath=self.database_infos[best_positive_index]['db_satellite_path'],split=self.split)  # [3,h,w]
        # positive_db_roadmap_map = load_dbimage(datapath=self.database_infos[best_positive_index]['db_roadmap_path'],split=self.split)  # [3,h,w]
        # positive_db_map = torch.stack([positive_db_satellite_map, positive_db_roadmap_map], 0)  # [nmap,3,h,w]

        # negative_db_satellite_map = [load_dbimage(datapath=self.database_infos[e]['db_satellite_path'],split=self.split) for e in neg_indexes] # [nneg,3,h,w]
        # negative_db_satellite_map = torch.stack(negative_db_satellite_map, 0)  # [nneg,3,h,w]
        # negative_db_roadmap_map = [load_dbimage(datapath=self.database_infos[e]['db_roadmap_path'],split=self.split) for e in neg_indexes] # [nneg,3,h,w]
        # negative_db_roadmap_map = torch.stack(negative_db_roadmap_map, 0)  # [nneg,3,h,w]
        # negative_db_map = torch.stack([negative_db_satellite_map, negative_db_roadmap_map], 1)  # [nneg,nmap,3,h,w]

        maptype = opt.maptype.split('_')
        positive_db_map = []
        negative_db_map = []
        for each_maptype in maptype:
            if each_maptype == 'satellite':
                each_positive_db_map = load_dbimage(datapath=self.database_infos[best_positive_index]['db_satellite_path'],split=self.split)
                each_negative_db_map = [load_dbimage(datapath=self.database_infos[e]['db_satellite_path'],split=self.split) for e in neg_indexes]
            elif each_maptype == 'roadmap':
                each_positive_db_map = load_dbimage(datapath=self.database_infos[best_positive_index]['db_roadmap_path'],split=self.split)
                each_negative_db_map = [load_dbimage(datapath=self.database_infos[e]['db_roadmap_path'],split=self.split) for e in neg_indexes]
            each_negative_db_map = torch.stack(each_negative_db_map, 0)  # [nneg,3,h,w]
            positive_db_map.append(each_positive_db_map)
            negative_db_map.append(each_negative_db_map)
        positive_db_map = torch.stack(positive_db_map, 0)  # [nmap,3,h,w]
        negative_db_map = torch.stack(negative_db_map, 1)  # [nneg,nmap,3,h,w]
        positive_db_eastnorth = torch.tensor([self.database_infos[best_positive_index]['east'], self.database_infos[best_positive_index]['north']]).float()
        negative_db_eastnorth = torch.tensor([[self.database_infos[e]['east'], self.database_infos[e]['north']] for e in neg_indexes]).float()



        # negative_db_map = torch.stack(negative_db_map, 0)  # [nneg,3,h,w]
        db_map = torch.cat((positive_db_map.unsqueeze(0), negative_db_map), 0)  # [1+nneg,nmap,3,h,w]
        db_eastnorth = torch.cat((positive_db_eastnorth.unsqueeze(0), negative_db_eastnorth), 0)  # [1+nneg,2]

        

        triplets_local_indexes = torch.empty((0, 3), dtype=torch.int)
        for neg_num in range(len(neg_indexes)):
            triplets_local_indexes = torch.cat((triplets_local_indexes, torch.tensor([0, 1, 2 + neg_num]).reshape(1, 3)))
        assert query_index < self.queries_num   

        output_dict = {
            'query_image': query_image,
            'query_bev': query_bev,
            'query_sph': query_sph,
            'query_pc': query_pc,
            'query_eastnorth': query_eastnorth,
            'db_map': db_map,
            'db_eastnorth': db_eastnorth,
        }

        return output_dict, triplets_local_indexes, self.triplets_global_indexes[index]
    
    def __len__(self):
        if self.is_inference:
            # At inference time return the number of images. This is used for caching or computing NetVLAD's clusters
            length = super().__len__()
            return length
        else:
            length = len(self.triplets_global_indexes)
            return length
    
    def compute_triplets(self, args, model, modelq=None):
        self.is_inference = True
        if self.mining == "full":
            self.compute_triplets_full(args, model)
        elif self.mining == "partial" or self.mining == "msls_weighted":
            self.compute_triplets_partial(args, model)
        elif self.mining == "random":
            self.compute_triplets_random(args, model)
        elif self.mining == 'partial_sep':
            assert modelq is not None
            self.compute_triplets_partial_sep(args, model, modelq)
        else:
            raise NotImplementedError
    
    @staticmethod
    def compute_cache(args, model, subset_ds, cache_shape):
        """Compute the cache containing features of images, which is used to
        find best positive and hardest negatives."""
        subset_dl = DataLoader(dataset=subset_ds, num_workers=args.num_workers,
                               batch_size=args.infer_batch_size, shuffle=False,
                               pin_memory=(args.device == "cuda"))
        model = model.eval()
        
        # RAMEfficient2DMatrix can be replaced by np.zeros, but using
        # RAMEfficient2DMatrix is RAM efficient for full database mining.
        cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32) # [db+q, c]
        with torch.no_grad():
            for images, indexes in tqdm(subset_dl):
                images = images.to(args.device)
                features = model(images)
                cache[indexes.numpy()] = features.cpu().numpy()
        return cache
    


    # =================== compute cache separate
    @staticmethod
    def compute_cache_sep(args, model, subset_ds, cache_shape, modelq):
        """Compute the cache containing features of images, which is used to
        find best positive and hardest negatives."""
        # subset_dl = DataLoader(dataset=subset_ds, num_workers=args.num_workers,
        #                        batch_size=args.infer_batch_size, shuffle=False,
        #                        pin_memory=(args.device == "cuda"))
        model = model.eval()
        modelq = modelq.eval()

        subset_ds_db = Subset(subset_ds.dataset, list(range(subset_ds.dataset.database_num)))
        subset_ds_q = Subset(subset_ds.dataset, list(range(subset_ds.dataset.database_num, len(subset_ds.dataset))))
        subset_dl_db = DataLoader(dataset=subset_ds_db, num_workers=args.num_workers,
                                 batch_size=args.infer_batch_size, shuffle=False,
                                 pin_memory=(args.device == "cuda"),
                                 collate_fn=nuscenes_collate_fn_cache_db
                                 )
        subset_dl_q = DataLoader(dataset=subset_ds_q, num_workers=args.num_workers,
                                batch_size=args.infer_batch_size, shuffle=False,
                                pin_memory=(args.device == "cuda"),
                                collate_fn=nuscenes_collate_fn_cache_q
                                )
        # RAMEfficient2DMatrix can be replaced by np.zeros, but using
        # RAMEfficient2DMatrix is RAM efficient for full database mining.
        cache = RAMEfficient2DMatrix(cache_shape, dtype=np.float32) # [db+q, c]
        with torch.no_grad():
            for data_dict, indexes in tqdm(subset_dl_db):
                for _k, _v in data_dict.items():
                    if isinstance(_v, torch.Tensor): data_dict[_k] = _v.to(args.device)
                # data_dict = {k: v.to(args.device) for k, v in data_dict.items()}
                features = model(data_dict, mode='db')
                cache[indexes.numpy()] = features['embedding'].cpu().numpy()
            for data_dict, indexes in tqdm(subset_dl_q):
                for _k, _v in data_dict.items():
                    if isinstance(_v, torch.Tensor): data_dict[_k] = _v.to(args.device)
                # data_dict = {k: v.to(args.device) for k, v in data_dict.items()}
                features = modelq(data_dict, mode='q')
                cache[indexes.numpy()] = features['embedding'].cpu().numpy()
        return cache
    



    def get_query_features(self, query_index, cache):
        query_features = cache[query_index + self.database_num]
        if query_features is None:
            raise RuntimeError(f"For query {self.queries_paths[query_index]} " +
                               f"with index {query_index} features have not been computed!\n" +
                               "There might be some bug with caching")
        return query_features
    
    def get_best_positive_index(self, args, query_index, cache, query_features):
        positives_features = cache[self.hard_positives_per_query[query_index]]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(positives_features)
        # Search the best positive (within 10 meters AND nearest in features space)
        _, best_positive_num = faiss_index.search(query_features.reshape(1, -1), 1)
        best_positive_index = self.hard_positives_per_query[query_index][best_positive_num[0]].item()
        return best_positive_index
    
    def get_hardest_negatives_indexes(self, args, cache, query_features, neg_samples):
        neg_features = cache[neg_samples]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(neg_features)
        # Search the 10 nearest negatives (further than 25 meters and nearest in features space)
        _, neg_nums = faiss_index.search(query_features.reshape(1, -1), self.negs_num_per_query)
        neg_nums = neg_nums.reshape(-1)
        neg_indexes = neg_samples[neg_nums].astype(np.int32)
        return neg_indexes
    


    # =================== random

    def compute_triplets_random(self, args, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False)
        # Take all the positives
        positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes]
        positives_indexes = [p for pos in positives_indexes for p in pos]  # Flatten list of lists to a list
        positives_indexes = list(np.unique(positives_indexes))
        
        # Compute the cache only for queries and their positives, in order to find the best positive
        subset_ds = Subset(self, positives_indexes + list(sampled_queries_indexes + self.database_num))
        cache = self.compute_cache(args, model, subset_ds, (len(self), args.features_dim))
        
        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            
            # Choose some random database images, from those remove the soft_positives, and then take the first 10 images as neg_indexes
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.random.choice(self.database_num, size=self.negs_num_per_query+len(soft_positives), replace=False)
            neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)[:self.negs_num_per_query]
            
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)
    


    # =================== full

    def compute_triplets_full(self, args, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False)
        # Take all database indexes
        database_indexes = list(range(self.database_num))
        #  Compute features for all images and store them in cache
        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))
        cache = self.compute_cache(args, model, subset_ds, (len(self), args.features_dim))
        
        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            # Choose 1000 random database images (neg_indexes)
            neg_indexes = np.random.choice(self.database_num, self.neg_samples_num, replace=False)
            # Remove the eventual soft_positives from neg_indexes
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(neg_indexes, soft_positives, assume_unique=True)
            # Concatenate neg_indexes with the previous top 10 negatives (neg_cache)
            neg_indexes = np.unique(np.concatenate([self.neg_cache[query_index], neg_indexes]))
            # Search the hardest negatives
            neg_indexes = self.get_hardest_negatives_indexes(args, cache, query_features, neg_indexes)
            # Update nearest negatives in neg_cache
            self.neg_cache[query_index] = neg_indexes
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)
    



    # =================== partial

    def compute_triplets_partial(self, args, model):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        if self.mining == "partial":
            sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False) 
        elif self.mining == "msls_weighted":  # Pick night and sideways queries with higher probability
            sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False, p=self.weights)
        
        # Sample 1000 random database images for the negatives
        sampled_database_indexes = np.random.choice(self.database_num, self.neg_samples_num, replace=False) # why need this step?
        # Take all the positives
        positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes] # [array, array, ...]
        positives_indexes = [p for pos in positives_indexes for p in pos] # [int, int, ...]
        # Merge them into database_indexes and remove duplicates
        database_indexes = list(sampled_database_indexes) + positives_indexes
        database_indexes = list(np.unique(database_indexes))
        
        # self is inference True length = 23949
        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))
        # [partial dataset + partial query] cache
        # query_index = query_index + self.database_num
        cache_length = len(self)
        cache = self.compute_cache(args, model, subset_ds, cache_shape=(cache_length, args.features_dim)) 
        
        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes):
            query_features = self.get_query_features(query_index, cache) # query_features = cache[query_index + self.database_num]
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            
            # Choose the hardest negatives within sampled_database_indexes, ensuring that there are no positives
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(sampled_database_indexes, soft_positives, assume_unique=True)
            
            # Take all database images that are negatives and are within the sampled database images (aka database_indexes)
            neg_indexes = self.get_hardest_negatives_indexes(args, cache, query_features, neg_indexes)
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)



    # =================== partial_sep 

    def compute_triplets_partial_sep(self, args, model, modelq):
        self.triplets_global_indexes = []
        # Take 1000 random queries
        if self.mining in ["partial", 'partial_sep']:
            sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False) 
        elif self.mining == "msls_weighted":  # Pick night and sideways queries with higher probability
            sampled_queries_indexes = np.random.choice(self.queries_num, args.cache_refresh_rate, replace=False, p=self.weights)
        
        # Sample 1000 random database images for the negatives
        sampled_database_indexes = np.random.choice(self.database_num, self.neg_samples_num, replace=False) # why need this step?
        # Take all the positives
        positives_indexes = [self.hard_positives_per_query[i] for i in sampled_queries_indexes] # [array, array, ...]
        positives_indexes = [p for pos in positives_indexes for p in pos] # [int, int, ...]
        # Merge them into database_indexes and remove duplicates
        database_indexes = list(sampled_database_indexes) + positives_indexes
        database_indexes = list(np.unique(database_indexes))
        
        # self is inference True length = 23949
        subset_ds = Subset(self, database_indexes + list(sampled_queries_indexes + self.database_num))
        # [partial dataset + partial query] cache
        # query_index = query_index + self.database_num
        cache_length = len(self)
        # cache = self.compute_cache(args, model, subset_ds, cache_shape=(cache_length, args.features_dim)) 
        cache = self.compute_cache_sep(args, model, subset_ds, cache_shape=(cache_length, args.features_dim), modelq=modelq) 
        
        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes): # max-3911  min-1
            query_features = self.get_query_features(query_index, cache) # query_features = cache[query_index + self.database_num]
            best_positive_index = self.get_best_positive_index(args, query_index, cache, query_features)
            
            # Choose the hardest negatives within sampled_database_indexes, ensuring that there are no positives
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(sampled_database_indexes, soft_positives, assume_unique=True)
            
            # Take all database images that are negatives and are within the sampled database images (aka database_indexes)
            neg_indexes = self.get_hardest_negatives_indexes(args, cache, query_features, neg_indexes)
            self.triplets_global_indexes.append((query_index, best_positive_index, *neg_indexes))
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(self.triplets_global_indexes)









class RAMEfficient2DMatrix:
    """This class behaves similarly to a numpy.ndarray initialized
    with np.zeros(), but is implemented to save RAM when the rows
    within the 2D array are sparse. In this case it's needed because
    we don't always compute features for each image, just for few of
    them"""
    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.matrix = [None] * shape[0]
    
    def __setitem__(self, indexes, vals):
        assert vals.shape[1] == self.shape[1], f"{vals.shape[1]} {self.shape[1]}"
        for i, val in zip(indexes, vals):
            self.matrix[i] = val.astype(self.dtype, copy=False)
    
    def __getitem__(self, index):
        if hasattr(index, "__len__"):
            return np.array([self.matrix[i] for i in index])
        else:
            return self.matrix[index]
