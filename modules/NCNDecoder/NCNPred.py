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
        self.xslin = nn.Linear(hidden_channels, out_channels)
        
        '''
        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(), nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU())
        self.xcnlin = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                nn.ReLU(), nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU())
        self.xijlini = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                nn.ReLU(), nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU())
        self.xijlinj = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                nn.ReLU(), nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU())
        self.xijfinal = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                nn.ReLU(), nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(), nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU())
        self.xslin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(), nn.Linear(hidden_channels, out_channels),
                nn.ReLU())
                '''
        '''
        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, out_channels))
        '''
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
        x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        xcn = spmm_add(cn, x)

        #print("xcns", xcns)
        #import time
        #time.sleep(2)
        xij = self.xijlini(xi) + self.xijlinj(xj)
        xcn = self.xcnlin(xcn) * self.beta
        xs = torch.cat([xij, xcn])
        xs.relu()
        xs = self.xslin(xs)
        '''
        if boolen:
            res = xs
        else:
            res = -xs
        '''

        return xs

    def forward(self, x, adj, tar_ei, boolen, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, boolen, filled1, [])
