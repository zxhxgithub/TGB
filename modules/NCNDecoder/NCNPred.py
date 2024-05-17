### 230913 #########################
import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F

from typing import Final, Iterable
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add

from modules.NCNDecoder.utils import adjoverlap, DropAdj

class NCNPredictor(torch.nn.Module):
    cndeg: Final[int]
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 dropout=0.3,
                 edrop=0.0,
                 ln=False,
                 cndeg=-1,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False,
                 beta=1.0):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Linear(hidden_channels, hidden_channels)
        self.xcnlin = nn.Linear(in_channels, hidden_channels)
        self.xijlini = nn.Linear(in_channels, hidden_channels)
        self.xijlinj = nn.Linear(in_channels, hidden_channels)
        self.xijfinal = nn.Linear(in_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        
        self.xslin = nn.Linear(2*in_channels, out_channels)

        self.cndeg = cndeg

    def multidomainforward(self,
                           x,
                           adj,
                           tar_ei,
                           boolen,
                           filled1: bool = False,
                           cndropprobs: Iterable[float] = []):
        
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        # x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        # print(cn)
        xcn = spmm_add(cn, x)
        # print(xcn)

        # xij = self.xijlini(xi) + self.xijlinj(xj)
        # xcn = self.xcnlin(xcn) * self.beta
        xij = torch.mul(xi, xj).reshape(-1, x.size(1))
        xs = torch.cat([xij, xcn], dim=-1)
        xs.relu()
        xs = self.xslin(xs)

        return xs

    def forward(self, x, adj, tar_ei, boolen, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, boolen, filled1, [])
