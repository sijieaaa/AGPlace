



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as TVM



import MinkowskiEngine as ME
from layers.eca_block import ECABasicBlock
from layers.pooling import MinkGeM

from tools.options import parse_arguments
opt = parse_arguments()








def ME_broadcast_add(sptensor, vec):
    assert isinstance(sptensor, ME.SparseTensor)
    assert isinstance(vec, torch.Tensor)
    vec_sp = ME.SparseTensor(vec, coordinate_map_key=sptensor.coordinate_map_key, 
                             coordinate_manager=sptensor.coordinate_manager)
    output = ME.MinkowskiBroadcastAddition()(sptensor, vec_sp)
    return output

def ME_broadcast_mul(sptensor, vec):
    assert isinstance(sptensor, ME.SparseTensor)
    assert isinstance(vec, torch.Tensor)
    vec_sp = ME.SparseTensor(vec, coordinate_map_key=sptensor.coordinate_map_key, 
                             coordinate_manager=sptensor.coordinate_manager)
    output = ME.MinkowskiBroadcastMultiplication()(sptensor, vec_sp)
    return output


def select_act(act):
    if act is None:
        out = nn.Identity()
    elif act == None:
        out = nn.Identity()
    elif act == 'relu':
        out = nn.ReLU()
    elif act == 'tanh':
        out = nn.Tanh()
    elif act == 'sigmoid':
        out = nn.Sigmoid()
    else:
        raise NotImplementedError
    return out




class BasicBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class Basic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out += identity
        out = self.relu(out)
        return out


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: [b, c, h, w]
        assert len(x.shape) == 4
        x = nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
        assert len(x.shape) == 4
        return x


class FFNFuse(nn.Module):
    def __init__(self, dim, stg2fuse_type):
        super().__init__()
        self.stg2fuse_type = stg2fuse_type.split('_')
        self.ffns = nn.ModuleList()
        for e in self.stg2fuse_type:
            if e == None:
                None
            elif e == 'basic':
                self.ffns.append(Basic(dim))
            else:
                raise NotImplementedError
    def forward(self, x):
        outlist = []
        for ffn in self.ffns:
            out = ffn(x)
            outlist.append(out)
        outsum = sum(outlist)
        return outsum
                                                                                                 


class Stage2FuseBlockAdd(nn.Module):
    def __init__(self, fusedim, imgdim, bevdim, voxdim):
        super().__init__()
        
        self.projsfusebev = nn.ModuleList()
        self.projsfuseimg = nn.ModuleList()
        self.projsfusevox = nn.ModuleList()
        self.ffnsbev = nn.ModuleList()
        self.ffnsimg = nn.ModuleList()
        self.ffnsvox = nn.ModuleList()
        self.projsbevfuse = nn.ModuleList()
        self.projsimgfuse = nn.ModuleList()
        self.projsvoxfuse = nn.ModuleList()
        self.ffnsfuse = nn.ModuleList()
        for i in range(opt.stg2nlayers):
            if opt.stg2_useproj == True:
                self.projsfuseimg.append(nn.Sequential(
                    nn.Linear(fusedim, imgdim)))
                self.projsfusevox.append(nn.Sequential(
                    nn.Linear(fusedim, voxdim)))
                self.projsimgfuse.append(nn.Sequential(
                    nn.Conv2d(imgdim, fusedim, kernel_size=1)))
                self.projsvoxfuse.append(nn.Sequential(
                    ME.MinkowskiConvolution(voxdim, fusedim, kernel_size=1, dimension=3)))
            else:
                self.projsfusebev.append(nn.Identity())
                self.projsfuseimg.append(nn.Identity())
                self.projsfusevox.append(nn.Identity())
                self.projsbevfuse.append(nn.Identity())
                self.projsimgfuse.append(nn.Identity())
                self.projsvoxfuse.append(nn.Identity())
            self.ffnsimg.append(BasicBlock(imgdim))
            self.ffnsvox.append(ECABasicBlock(voxdim, voxdim))
            self.ffnsfuse.append(FFNFuse(dim=fusedim, stg2fuse_type=opt.stg2fuse_type))


        # self.poolbev = GeM()
        self.poolimage = GeM()
        self.poolvox = MinkGeM()
        self.poolfuse = GeM()

    def forward_imgvox(self, imgmap, bevmap, voxmap, fusevec):
        # imagefeatmap: [b, c, h, w]
        # bevfeatmap: [b, c, h, w]
        # fusevec: [b, c]
        for i in range(opt.stg2nlayers):
            projfuseimg = self.projsfuseimg[i]
            projfusevox = self.projsfusevox[i]
            ffnimg = self.ffnsimg[i]
            ffnvox = self.ffnsvox[i]
            projimgfuse = self.projsimgfuse[i]
            projvoxfuse = self.projsvoxfuse[i]
            ffnfuse = self.ffnsfuse[i]

            if opt.stg2_type == 'full':
                fusevec_img = projfuseimg(fusevec)
                fusevec_vox = projfusevox(fusevec)
                imgmap = imgmap + fusevec_img.unsqueeze(-1).unsqueeze(-1)
                voxmap = ME_broadcast_add(voxmap, fusevec_vox)

                imgmap = ffnimg(imgmap)
                # bevmap = ffnbev(bevmap)
                voxmap = ffnvox(voxmap)
                imgoutvec = self.poolimage(imgmap).flatten(1) 
                voxoutvec = self.poolvox(voxmap).flatten(1)

                if opt.stg2fuse_type == None:
                    None
                else:
                    imgmap_fuse = projimgfuse(imgmap)
                    voxmap_fuse = projvoxfuse(voxmap)
                    imgvec_fuse = F.adaptive_avg_pool2d(imgmap_fuse, [1,1]).squeeze(-1).squeeze(-1)
                    voxvec_fuse = ME.MinkowskiGlobalAvgPooling()(voxmap_fuse) 
                    fusevec = fusevec + imgvec_fuse + voxvec_fuse.F
                    fusevec = ffnfuse(fusevec)


            else:
                raise NotImplementedError

        return fusevec, imgoutvec, None, voxoutvec
    


    
    def forward_imgbev(self, imagemap, bevmap, voxmap, fusevec):
        # imagefeatmap: [b, c, h, w]
        # bevfeatmap: [b, c, h, w]
        # fusevec: [b, c]
        # assert imagemap.shape[1] == bevmap.shape[1]
        # assert imagemap.shape[1] == fusevec.shape[1]
        for i in range(opt.stg2nlayers):
            projfuseimage = self.projsfuseimg[i]
            projfusebev = self.projsfusebev[i]
            # projfusevox = self.projsfusevox[i]
            ffnimage = self.ffnsimg[i]
            ffnbev = self.ffnsbev[i]
            # ffnvox = self.ffnsvox[i]
            projimgfuse = self.projsimgfuse[i]
            projbevfuse = self.projsbevfuse[i]
            # projvoxfuse = self.projsvoxfuse[i]
            ffnfuse = self.ffnsfuse[i]

            if opt.stg2_type == 'full':
                fusevec_image = projfuseimage(fusevec)
                fusevec_bev = projfusebev(fusevec)
                # fusevec_vox = projfusevox(fusevec)
                imagemap = imagemap + fusevec_image.unsqueeze(-1).unsqueeze(-1)
                bevmap = bevmap + fusevec_bev.unsqueeze(-1).unsqueeze(-1)
                # voxmap = ME_broadcast_add(voxmap, fusevec_vox)

                imagemap = ffnimage(imagemap)
                bevmap = ffnbev(bevmap)
                # voxmap = ffnvox(voxmap)
                imageoutvec = self.poolimage(imagemap).flatten(1) 
                bevoutvec = self.poolbev(bevmap).flatten(1)
                # voxoutvec = self.poolvox(voxmap).flatten(1)

                if opt.stg2fuse_type == 'res':
                    None
                else:
                    imgmap_fuse = projimgfuse(imagemap)
                    bevmap_fuse = projbevfuse(bevmap)
                    # voxmap_fuse = projvoxfuse(voxmap)
                    imgvec_fuse = F.adaptive_avg_pool2d(imgmap_fuse, [1,1]).squeeze(-1).squeeze(-1)
                    bevvec_fuse = F.adaptive_avg_pool2d(bevmap_fuse, [1,1]).squeeze(-1).squeeze(-1)
                    # voxvec_fuse = ME.MinkowskiGlobalAvgPooling()(voxmap_fuse) 
                    fusevec = fusevec + imgvec_fuse + bevvec_fuse 
                    # fusevec = fusevec + imagevec_fuse + voxvec_fuse.F
                    fusevec = ffnfuse(fusevec)

            elif opt.stg2_type == 'image_bev':
                imagemap = ffnimage(imagemap)
                bevmap = ffnbev(bevmap)
                imageoutvec = self.poolimage(imagemap).flatten(1) 
                bevoutvec = self.poolbev(bevmap).flatten(1)
            else:
                raise NotImplementedError

        return fusevec, imageoutvec, bevoutvec, None
        # return fusevec, imageoutvec, None, voxoutvec
    





    def forward(self, imagemap, bevmap, voxmap, fusevec, type):
        if type == 'vox':
            fusevec, imageoutvec, bevoutvec, voxoutvec = self.forward_imgvox(imagemap, bevmap, voxmap, fusevec)
        elif type == 'bev':
            fusevec, imageoutvec, bevoutvec, voxoutvec = self.forward_imgbev(imagemap, bevmap, voxmap, fusevec)
        else:
            raise NotImplementedError
        return fusevec, imageoutvec, bevoutvec, voxoutvec