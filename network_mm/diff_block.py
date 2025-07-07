




import torch.nn as nn
from network_mm.ffns import FCODE



from tools.options import parse_arguments
opt = parse_arguments()





class DiffBlock(nn.Module):
    def __init__(self, dim, ode_dim):
        super().__init__()
        self.blocks = nn.ModuleList()

        diff_type = opt.diff_type


        for e in diff_type.split('_'):
            e, act = e.split('@')
            if e == None:
                None
            elif e == 'fcode':
                self.blocks.append(FCODE(dim,act))
            else:
                raise NotImplementedError

    def forward(self, x, z0=None):
        # input: [b,n,c]

        # identity = x
        outlist = []
        for block in self.blocks:
            if z0 is not None: # for CDE
                out = block(x, z0=z0)
            else: 
                out = block(x)
            outlist.append(out)
        
        out = sum(outlist)

        return out