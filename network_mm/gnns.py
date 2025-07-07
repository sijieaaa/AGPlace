

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint

from tools.options import parse_arguments
opt = parse_arguments()


class ODEFunc(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, t, x):
        output = self.func(x)
        return output



# ==== QKV attention
    
class QKVAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim, dim)
        self.fc_k = nn.Linear(dim, dim)
        self.fc_v = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, N, C]
        q = self.fc_q(x) # [B, N, C]
        k = self.fc_k(x)
        v = self.fc_v(x)

        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.dim // self.num_heads) # [B, N, H, C//H]
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.dim // self.num_heads)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.dim // self.num_heads)

        q = q.permute(0, 2, 1, 3) # [B, H, N, C//H]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1) # [B, H, N, N]
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v # [B, H, N, C//H]
        out = out.permute(0, 2, 1, 3).contiguous() # [B, N, H, C//H]
        out = out.view(out.shape[0], out.shape[1], self.dim) # [B, N, C]
        
        return out
    







# ==== beltrami_grand

class Beltrami(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.fc = nn.Linear(dim, dim*2)
        self.k = k

    def forward(self, x):
        assert len(x.shape) == 3
        b,n,c = x.shape
        feat_pos = self.fc(x) # [b,hw,c*2]
        feat = feat_pos[:,:,:c] # [b,hw,c]
        pos = feat_pos[:,:,c:] # [b,hw,c]
        pos = nn.functional.normalize(pos, p=2, dim=-1)
        sim = pos @ pos.transpose(-1,-2) # [b,hw,hw]
        topksim, topkid = torch.topk(sim, k=self.k, dim=-1) # [b,hw,k]
        # fetch topk feat
        topkid = topkid.flatten(1) # [b,hw*k] 
        topkfeat = torch.gather(feat, dim=1, index=topkid.unsqueeze(-1).expand(-1,-1,c))
        topkfeat = topkfeat.view(b,n,self.k,c) # [b,hw,k,c]
        attn = topksim.softmax(dim=-1) # [b,hw,k]
        x = (attn.unsqueeze(-1) * topkfeat).sum(dim=-2) # [b,hw,c]

        return x
    

class BeltramiODE(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        func = Beltrami(dim=dim, k=k)
        self.odefunc = ODEFunc(func)

    def forward(self, x):
        t = torch.tensor([0,1], dtype=torch.float32, device=x.device, requires_grad=False)
        output = odeint_adjoint(self.odefunc, x, t, 
                        method=opt.odeint_method,
                        rtol=opt.tol, atol=opt.tol)
        output = output[-1]

        return output
