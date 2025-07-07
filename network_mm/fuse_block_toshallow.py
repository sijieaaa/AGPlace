import torch
import torch.nn as nn
import torch.nn.functional as F
from network_mm.diff_block import DiffBlock
import MinkowskiEngine as ME

from tools.options import parse_arguments
opt = parse_arguments()

class FuseBlockToShallow(nn.Module):
    def __init__(self, dims=[256,256,256], img_dims=[64,128,256], vox_dims=[64,128,256], bev_dims=[64,128,256]):
        super().__init__()

        self.dims = dims
        self.img_dims = img_dims
        self.vox_dims = vox_dims
        self.bev_dims = bev_dims
        self.blocks = nn.ModuleList()
        self.updimsbev = nn.ModuleList()
        self.updimsimg = nn.ModuleList()
        self.updimsvox = nn.ModuleList()
        for i in range(len(dims)):
            diffblock = DiffBlock(dim=dims[-1], ode_dim=dims[-1])
            self.blocks.append(diffblock)
            if i < len(dims)-1:
                self.updimsimg.append(nn.Linear(self.img_dims[i], dims[-1]))
                self.updimsvox.append(nn.Linear(self.vox_dims[i], dims[-1]))
            else:
                self.updimsimg.append(nn.Identity())
                self.updimsvox.append(nn.Identity())
            
        # self.cde = DiffBlock(dim=dims[-1], ode_dim=dims[-1])

    def forward_imgbev(self, imagemaplist, bevmaplist=None, voxmaplist=None):
        assert len(imagemaplist) == len(self.dims)

        imageveclist = [F.adaptive_avg_pool2d(e, output_size=1).flatten(1) for e in imagemaplist] 
        bevveclist = [F.adaptive_avg_pool2d(e, output_size=1).flatten(1) for e in bevmaplist]

        if 'cde' in opt.diff_type:
            # ==== cde
            if opt.diff_direction == 'forward':
                imageveclist = [self.updimsimg[i](imageveclist[i]) for i in range(len(imageveclist))]
                bevveclist = [self.updimsbev[i](bevveclist[i]) for i in range(len(bevveclist))]
            elif opt.diff_direction == 'backward':
                imageveclist = [self.updimsimg[i](imageveclist[i]) for i in range(len(imageveclist)-1,-1,-1)]
                bevveclist = [self.updimsbev[i](bevveclist[i]) for i in range(len(bevveclist)-1,-1,-1)]
            imageveclist = torch.stack(imageveclist, dim=1) # [b,seq,c]
            bevveclist = torch.stack(bevveclist, dim=1)
            fuseveclist = imageveclist + bevveclist
            fusevec = self.cde(fuseveclist,z0=fuseveclist[:,0])
        else:
            # ==== deep to shallow
            fusevec = 0 
            for i in range(len(self.dims)):
                if opt.diff_direction == 'forward':
                    i = i
                elif opt.diff_direction == 'backward':
                    i = len(self.dims)-1-i
                imagevec = imageveclist[i]
                bevvec = bevveclist[i]
                block = self.blocks[i]
                updimimage = self.updimsimg[i]
                updimbev = self.updimsbev[i]

                imagevec = updimimage(imagevec)
                bevvec = updimbev(bevvec)

                fusevec = fusevec + imagevec + bevvec   
                fusevec = block(fusevec)

        return fusevec
    





    def forward_imgvox(self, imagemaplist, bevmaplist=None, voxmaplist=None):
        assert len(imagemaplist) == len(self.dims)

        imageveclist = [F.adaptive_avg_pool2d(e, output_size=1).flatten(1) for e in imagemaplist] 
        voxveclist = [ME.MinkowskiGlobalPooling()(e).F for e in voxmaplist]

        if 'cde' in opt.diff_type:
            # ==== cde
            if opt.diff_direction == 'forward':
                imageveclist = [self.updimsimg[i](imageveclist[i]) for i in range(len(imageveclist))]
                voxveclist = [self.updimsvox[i](voxveclist[i]) for i in range(len(voxveclist))]
            elif opt.diff_direction == 'backward':
                imageveclist = [self.updimsimg[i](imageveclist[i]) for i in range(len(imageveclist)-1,-1,-1)]
                voxveclist = [self.updimsvox[i](voxveclist[i]) for i in range(len(voxveclist)-1,-1,-1)]
            imageveclist = torch.stack(imageveclist, dim=1) # [b,seq,c]
            voxveclist = torch.stack(voxveclist, dim=1)
            fuseveclist = imageveclist + voxveclist
            fusevec = self.cde(fuseveclist,z0=fuseveclist[:,0])

        else:
            # ==== deep to shallow
            fusevec = 0 
            for i in range(len(self.dims)):
                if opt.diff_direction == 'forward':
                    i = i
                elif opt.diff_direction == 'backward':
                    i = len(self.dims)-1-i
                imagevec = imageveclist[i]
                voxvec = voxveclist[i]
                block = self.blocks[i]
                updimimage = self.updimsimg[i]
                updimvox = self.updimsvox[i]

                imagevec = updimimage(imagevec)
                voxvec = updimvox(voxvec)

                fusevec = fusevec + imagevec + voxvec  
                fusevec = block(fusevec)




        return fusevec
    




    def forward(self, imagefeatmaplist, bevfeatmaplist, voxfeatmaplist, type=None):
        if type == 'bev':
            output = self.forward_imgbev(imagefeatmaplist, bevfeatmaplist, voxfeatmaplist)
        elif type == 'vox':
            output = self.forward_imgvox(imagefeatmaplist, bevfeatmaplist, voxfeatmaplist)
        else:
            raise NotImplementedError
        return output