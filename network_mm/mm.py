









import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from network_mm.image_fe import ImageFE
from network_mm.image_pooling import GeM
from network_mm.fuse_block_toshallow import FuseBlockToShallow
from network_mm.stage2fuse_blockadd import Stage2FuseBlockAdd

from models.minkfpn import MinkFPN
from layers.pooling import MinkGeM 
from layers.eca_block import ECABasicBlock

import MinkowskiEngine as ME

from tools.options import parse_arguments
opt = parse_arguments()

class MM(nn.Module):
    def __init__(self , drop=None):
        super().__init__()
        self.drop = drop
        # ---- query
        self.image_fe = ImageFE(fe_type=opt.mm_imgfe, layers=opt.mm_imgfe_layers)
        self.image_pool = GeM()
        planes = [int(x) for x in opt.mm_voxfe_planes.split('_')]
        layers = [int(x) for x in opt.mm_voxfe_layers.split('_')]
        self.vox_fe = MinkFPN(in_channels=1, out_channels=planes[-1], 
                              planes=planes, layers=layers,
                              num_top_down=opt.mm_voxfe_ntd, conv0_kernel_size=5, block=ECABasicBlock)
        self.vox_pool = MinkGeM()
        
        self.fuseblocktoshallow = FuseBlockToShallow(dims=[opt.mm_stg2fuse_dim for e in range(len(planes))],
                                                     img_dims=[int(e) for e in opt.mm_imgfe_planes.split('_')],
                                                     vox_dims=[int(e) for e in opt.mm_voxfe_planes.split('_')],
                                                     bev_dims=[int(e) for e in opt.mm_bevfe_planes.split('_')]) 
        # 96_192_384
        self.stg2fuseblock = Stage2FuseBlockAdd(fusedim=opt.mm_stg2fuse_dim, imgdim=opt.mm_imgfe_dim, bevdim=opt.mm_bevfe_dim, voxdim=opt.mm_voxfe_dim)
        self.stg2fusefc = nn.Linear(opt.mm_stg2fuse_dim, opt.mm_stg2fuse_dim)


        self.image_weight = nn.Parameter(torch.tensor(opt.image_weight, dtype=torch.float32), requires_grad=opt.image_learnweight)
        self.vox_weight = nn.Parameter(torch.tensor(opt.vox_weight, dtype=torch.float32), requires_grad=opt.vox_learnweight)
        self.shallow_weight = nn.Parameter(torch.tensor(opt.shallow_weight, dtype=torch.float32), requires_grad=opt.shallow_learnweight)


        self.imageorg_weight = nn.Parameter(torch.tensor(opt.imagevoxorg_weight, dtype=torch.float32), requires_grad=opt.imagevoxorg_learnweight)
        self.voxorg_weight = nn.Parameter(torch.tensor(opt.imagevoxorg_weight, dtype=torch.float32), requires_grad=opt.imagevoxorg_learnweight)
        self.shalloworg_weight = nn.Parameter(torch.tensor(opt.shalloworg_weight, dtype=torch.float32), requires_grad=opt.shalloworg_learnweight)

        self.stg2image_weight = nn.Parameter(torch.tensor(opt.stg2imagevox_weight, dtype=torch.float32), requires_grad=opt.stg2imagevox_learnweight)
        self.stg2vox_weight = nn.Parameter(torch.tensor(opt.stg2imagevox_weight, dtype=torch.float32), requires_grad=opt.stg2imagevox_learnweight)
        self.stg2fuse_weight = nn.Parameter(torch.tensor(opt.stg2fuse_weight, dtype=torch.float32), requires_grad=opt.stg2fuse_learnweight)
        


    # ====  query
    def forward_q(self, data_dict):
        if self.drop == 'image':
            data_dict['query_image'] = data_dict['query_image'] * 0
        elif self.drop == 'pc':
            data_dict['coords'][:,1:] = data_dict['coords'][:,1:] * 0
        
        image = data_dict['query_image']
        output = []
        if 'image' in opt.output_type:
            imagefeatmap, imagefeatmaplist = self.image_fe(image)
            imagefeatvec = self.image_pool(imagefeatmap)
            imagefeatvec = imagefeatvec.flatten(1)
            if opt.output_l2 is True:
                imagefeatvec = F.normalize(imagefeatvec, dim=-1)
            imagefeatvec_org = imagefeatvec
            output.append(imagefeatvec * self.image_weight)
        if 'vox' in opt.output_type:
            sptensor = ME.SparseTensor(features=data_dict['features'], coordinates=data_dict['coords'])
            voxfeatmap, voxfeatmaplist = self.vox_fe(sptensor)
            voxfeatvec = self.vox_pool(voxfeatmap)
            if opt.output_l2 is True:
                voxfeatvec = F.normalize(voxfeatvec, dim=-1)
            voxfeatvec_org = voxfeatvec
            output.append(voxfeatvec * self.vox_weight)
            a=1

            
        # ==== stage-1 fusion, ME
        if 'shallow' in opt.output_type:
            if 'vox' in opt.output_type:
                shallowfeatvec = self.fuseblocktoshallow(imagefeatmaplist, None, voxfeatmaplist, type='vox')
            shallowfeatvecorg = shallowfeatvec
            if opt.output_l2 is True:
                shallowfeatvec = F.normalize(shallowfeatvec, dim=-1)
            output.append(shallowfeatvec * self.shallow_weight)
        elif 'addorg' in opt.output_type:
            if 'vox' in opt.output_type:
                addorgvec = imagefeatvec_org + voxfeatvec_org
            shallowfeatvecorg = shallowfeatvec
            if opt.output_l2 is True:
                addorgvec = F.normalize(addorgvec, dim=-1)
            output.append(addorgvec * self.shallow_weight)

        
        # ==== stage-2 fusion, ME
        if 'vox' in opt.output_type:
            stg2fusevec, stg2imagevec, stg2bevvec, stg2voxvec = self.stg2fuseblock(imagefeatmap, None, voxfeatmap, output[-1],type='vox')

        stg2fusevec = self.stg2fusefc(stg2fusevec)



        # ==== final output
        finaloutput = []
        if 'imageorg' in opt.final_type:
            finaloutput.append(imagefeatvec_org * self.imageorg_weight)
        if 'voxorg' in opt.final_type:
            finaloutput.append(voxfeatvec_org * self.voxorg_weight)
        if 'shalloworg' in opt.final_type:
            finaloutput.append(shallowfeatvec * self.shalloworg_weight)
        if 'stg2image' in opt.final_type:
            finaloutput.append(stg2imagevec * self.stg2image_weight)
        if 'stg2vox' in opt.final_type:
            finaloutput.append(stg2voxvec * self.stg2vox_weight)
        if 'stg2fuse' in opt.final_type:
            finaloutput.append(stg2fusevec * self.stg2fuse_weight)

        if opt.final_fusetype == 'add':
            x = sum(finaloutput)
        elif opt.final_fusetype == 'cat':
            x = torch.cat(finaloutput, dim=-1)
        elif opt.final_fusetype == 'catadd':
            x = torch.cat(finaloutput[:-1], dim=-1)
            x = x + finaloutput[-1]

        if opt.final_l2 is True:
            x = F.normalize(x, dim=-1)

        

        output_dict = {
            'imagevec_org': imagefeatvec_org,
            'voxvec_org': voxfeatvec_org,
            'shallowvec_org': shallowfeatvecorg,
            'stg2fusevec': stg2fusevec,
            'stg2imagevec': stg2imagevec,
            'stg2voxvec': stg2voxvec,
            'embedding': x
        }
        
        return output_dict




    def forward(self, data_dict, mode):
        # resize img
        if mode == 'q':
            x = self.forward_q(data_dict)
        else:
            raise NotImplementedError
        
        return x