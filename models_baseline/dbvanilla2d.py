




import torch
import torch.nn as nn
from typing import List
import torchvision.models as TVM
from network.image_fe import ImageFE
from network.image_pooling import GeM
import torch.nn.functional as F

from tools.options import parse_arguments
opt = parse_arguments()

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )
    def forward(self, x):
        output = self.seq(x)
        return output


class DBVanilla2D(nn.Module):
    def __init__(self, mode:List[str], dim):
        super().__init__()
        # ---- database
        if mode == 'db':
            maptype = opt.maptype.split('_')
            self.dbimage_fes = [ImageFE(fe_type=opt.dbimage_fe, layers=opt.dbimage_fe_layers) for _ in range(len(maptype))]
            self.dbimage_pools = [GeM() for _ in range(len(maptype))]
            # self.dbimage_mlp = [nn.Linear(e.last_dim, dim) for e in self.dbimage_fes] # after pool, change dim
            self.dbimage_mlps = [MLP(e.last_dim, dim) for e in self.dbimage_fes] # after pool, change dim
            self.dbimage_fes = nn.ModuleList(self.dbimage_fes)
            self.dbimage_pools = nn.ModuleList(self.dbimage_pools)
            self.dbimage_mlps = nn.ModuleList(self.dbimage_mlps)






    def forward_db(self, data_dict):
        db_map = data_dict['db_map'] 
        if len(db_map.shape) == 5: # [b,nmap,3,h,w]  for caching/testing
            mode = 'cachetest'
            b, nmap, c ,h, w = db_map.shape 
            assert c == 3
            db_map = db_map.unsqueeze(1) # [b,1,nmap,3,h,w]
            ndb = 1
        elif len(db_map.shape) == 6: # [b,ndb,nmap,3,h,w]   for training
            mode = 'train'
            b, ndb, nmap, c, h, w = db_map.shape
            assert c == 3
        else:
            raise NotImplementedError
        db_map = db_map.permute(2,0,1,3,4,5).contiguous() # [nmap,b,ndb,3,h,w]
        out_dbmap = []
        out_dbvec = []
        for i in range(len(db_map)):
            dbmap_i = db_map[i].view(-1, c, h, w) # [b*ndb,3,h,w]
            if opt.share_dbfe == True:
                dbmap_i, _ = self.dbimage_fes[0](dbmap_i)
                dbvec_i = self.dbimage_pools[0](dbmap_i)
                dbvec_i = self.dbimage_mlps[0](dbvec_i)
            else:
                dbmap_i, _ = self.dbimage_fes[i](dbmap_i) 
                dbvec_i = self.dbimage_pools[i](dbmap_i) # [b*ndb,c]
                dbvec_i = self.dbimage_mlps[i](dbvec_i) # [b*ndb,dim]
            out_dbmap.append(dbmap_i)
            out_dbvec.append(dbvec_i)
        out_dbvec = torch.stack(out_dbvec, dim=1) # [b*ndb,nmap,c]
        # TODO: fusion
        if opt.output_l2 == True:
            out_dbvec = F.normalize(out_dbvec, p=2, dim=-1) 
        out_dbvec = torch.mean(out_dbvec, dim=1) # [b*ndb,c]
        out_dbvec = out_dbvec.view(b, ndb, -1) # [b,ndb,c]
        # ENDTOD
        b, ndb, c = out_dbvec.shape
        if mode == 'cachetest':
            assert ndb == 1
            out_dbvec = out_dbvec.view(b, c)
        elif mode == 'train':
            assert ndb > 1
        else:
            raise NotImplementedError
        
        if opt.final_l2 == True:
            out_dbvec = F.normalize(out_dbvec, p=2, dim=-1)
            
        output_dict = {
            'embedding': out_dbvec,
        }
        return output_dict





    def forward(self, data_dict, mode:List[str]):
        if mode=='q':
            x = self.forward_q(data_dict)
        elif mode=='db':
            x = self.forward_db(data_dict)
        else:
            raise NotImplementedError
        return x