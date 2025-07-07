# Author: Jacek Komorowski
# Warsaw University of Technology

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from models.minkfpn import MinkFPN
from models.minkfpn import GeneralMinkFPN

import layers.pooling as pooling
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from layers.eca_block import ECABasicBlock

from layers.pooling import MinkGeM as MinkGeM

# from tools.options import Options
# args = Options().parse()

# from tools.utils import set_seed
# set_seed(7)




class ExtraBlock(nn.Module):
    def __init__(self, in_features, num_heads, kernel_size, stride, dimension):
        super(ExtraBlock, self).__init__()

        self.in_features = in_features
        self.num_heads = num_heads


        self.seq = nn.Sequential(
            ME.MinkowskiConvolution(in_features, int(num_heads*in_features), kernel_size=kernel_size, stride=stride, dimension=dimension),
        )
        self.minkgem = MinkGeM(int(num_heads*in_features))


    def forward(self, x):
        x = self.seq(x)
        x = self.minkgem(x) # 
        x = x.view(-1, self.num_heads, self.in_features)

        return x










class MinkLoc(torch.nn.Module):
    def __init__(self, in_channels, feature_size, output_dim, planes, layers, num_top_down, conv0_kernel_size,
                 block='BasicBlock', pooling_method='GeM', linear_block=False, dropout_p=None):
        # block: Type of the network building block: BasicBlock or SEBasicBlock
        # add_linear_layers: Add linear layers at the end
        # dropout_p: dropout probability (None = no dropout)

        super().__init__()
        self.in_channels = in_channels
        self.feature_size = feature_size    # Size of local features produced by local feature extraction block
        self.output_dim = output_dim        # Dimensionality of the global descriptor produced by pooling layer
        self.block = block

        if block == 'BasicBlock':
            block_module = BasicBlock
        elif block == 'Bottleneck':
            block_module = Bottleneck
        elif block == 'ECABasicBlock':
            block_module = ECABasicBlock
        else:
            raise NotImplementedError('Unsupported network block: {}'.format(block))

        self.pooling_method = pooling_method
        self.linear_block = linear_block
        self.dropout_p = dropout_p

        if args.minkfpn == 'minkfpn':
            self.backbone = MinkFPN(in_channels=in_channels, out_channels=self.feature_size, num_top_down=num_top_down,
                                    conv0_kernel_size=conv0_kernel_size, block=block_module, layers=layers, planes=planes)
        elif args.minkfpn == 'generalminkfpn':
            self.backbone = GeneralMinkFPN(in_channels=in_channels, out_channels=self.feature_size, num_top_down=num_top_down,
                                    conv0_kernel_size=conv0_kernel_size, block=block_module, layers=layers, planes=planes)
        else:
            raise NotImplementedError('Unsupported network block: {}'.format(args.minkfpn))



        self.pooling = pooling.PoolingWrapper(pool_method=pooling_method, in_dim=self.feature_size,
                                              output_dim=output_dim)
        self.pooled_feature_size = self.pooling.output_dim      # Number of channels returned by pooling layer

        if self.dropout_p is not None:
            self.dropout = nn.Dropout(p=self.dropout_p)
        else:
            self.dropout = None

        if self.linear_block:
            # At least output_dim neurons in intermediary layer
            int_channels = self.output_dim
            self.linear = nn.Sequential(nn.Linear(self.pooled_feature_size, int_channels, bias=False),
                                        nn.BatchNorm1d(int_channels, affine=True),
                                        nn.ReLU(inplace=True), nn.Linear(int_channels, output_dim))
        else:
            self.linear = None




        self.extra_block = ExtraBlock(128, num_heads=8, kernel_size=7, stride=1, dimension=3)




    def forward(self, batch):
        # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
        x = ME.SparseTensor(features=batch['features'], coordinates=batch['coords'])
        x = self.backbone(x)

        



        x_feat = x
        
        # embedding_extra = self.extra_block(x)





        # x is (num_points, n_features) tensor
        assert x.shape[1] == self.feature_size, 'Backbone output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.feature_size)

        x = self.pooling(x)
        if x.dim() == 3 and x.shape[2] == 1:
            # Reshape (batch_size,
            x = x.flatten(1)

        assert x.dim() == 2, 'Expected 2-dimensional tensor (batch_size,output_dim). Got {} dimensions.'.format(x.dim())
        assert x.shape[1] == self.pooled_feature_size, 'Backbone output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.pooled_feature_size)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.linear is not None:
            x = self.linear(x)

        assert x.shape[1] == self.output_dim, 'Output tensor has: {} channels. Expected: {}'.format(x.shape[1], self.output_dim)
        # x is (batch_size, output_dim) tensor
        x_gem = x

        
        return {
            'embedding': x_gem,
            'x_feat': x_feat,
            'embedding_extra':None
            }








# class GeneralMinkLoc(torch.nn.Module):
#     def __init__(self, in_channels, feature_size, output_dim, planes, layers, num_top_down, conv0_kernel_size,
#                  block='BasicBlock', pooling_method='GeM', linear_block=False, dropout_p=None):
        
#         super().__init__()












# class GeneralPointNetLoc(torch.nn.Module):
#     def __init__(self, in_channels, feature_size, output_dim, planes, layers, num_top_down, conv0_kernel_size,
#                  block='BasicBlock', pooling_method='GeM', linear_block=False, dropout_p=None):
        
#         super().__init__()



