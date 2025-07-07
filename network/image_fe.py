




import torch.nn as nn
import torchvision


class ImageFE(nn.Module):
    def __init__(self, fe_type, layers):
        super().__init__()

        self.fe_type = fe_type
        layers = [int(x) for x in layers.split('_')]
        self.layers = layers

        if self.fe_type == 'resnet18':
            self.fe = torchvision.models.resnet18(pretrained=True)
            if len(self.layers) == 2:
                self.last_dim = 128
                self.fe.layer3 = nn.Identity()
                self.fe.layer4 = nn.Identity()
            elif len(self.layers) == 3:
                self.last_dim = 256
                self.fe.layer4 = nn.Identity()
            elif len(self.layers) == 4:
                self.last_dim = 512
            else:
                raise NotImplementedError
        
        elif self.fe_type == 'resnet34':
            self.fe = torchvision.models.resnet34(pretrained=True)
            if len(self.layers) == 2:
                self.last_dim = 128
                self.fe.layer3 = nn.Identity()
                self.fe.layer4 = nn.Identity()
            elif len(self.layers) == 3:
                self.last_dim = 256
                self.fe.layer4 = nn.Identity()
            elif len(self.layers) == 4:
                self.last_dim = 512
            else:
                raise NotImplementedError
            

        elif self.fe_type == 'resnet50':
            self.fe = torchvision.models.resnet50(pretrained=True)
            if len(self.layers) == 2:
                self.last_dim = 512
                self.fe.layer3 = nn.Identity()
                self.fe.layer4 = nn.Identity()
            elif len(self.layers) == 3:
                self.last_dim = 1024
                self.fe.layer4 = nn.Identity()
            elif len(self.layers) == 4:
                self.last_dim = 2048
            else:
                raise NotImplementedError
            

            
        elif self.fe_type == 'squeezenet10':
            self.fe = torchvision.models.squeezenet1_0(pretrained=True)
            self.squeezenet_fc = nn.Conv2d(512, 256, kernel_size=1)
            self.last_dim = 256
        elif self.fe_type == 'squeezenet11':
            self.fe = torchvision.models.squeezenet1_1(pretrained=True)
            self.squeezenet_fc = nn.Conv2d(512, 256, kernel_size=1)
            self.last_dim = 256
        
        
            
        elif self.fe_type == 'convnext_tiny':
            self.fe = torchvision.models.convnext_tiny(pretrained=True)
            if len(self.layers) == 2:
                self.last_dim = 192
            elif len(self.layers) == 3:
                self.last_dim = 384
            elif len(self.layers) == 4:
                self.last_dim = 768
            else:
                raise NotImplementedError
            # ==== remove last two blocks
            layers_list = list(self.fe.features.children())
            assert len(layers_list)==8
            if len(self.layers) == 3:
                layers_list = layers_list[:-2]
            elif len(self.layers) == 4:
                layers_list = layers_list
            else:
                raise NotImplementedError
            # ==== remove layers in each block
            for i in range(len(layers_list)):
                if i == 1: 
                    layers_list[i] = layers_list[i][:self.layers[0]]
                if i == 3: 
                    layers_list[i] = layers_list[i][:self.layers[1]]
                if i == 5: 
                    layers_list[i] = layers_list[i][:self.layers[2]]
                if i == 7:
                    layers_list[i] = layers_list[i][:self.layers[3]]
            self.fe.features = nn.Sequential(*layers_list)

        else:
            raise NotImplementedError
        




    def forward_resnet(self, x):
        x = self.fe.conv1(x)
        x = self.fe.bn1(x)
        x = self.fe.relu(x)
        x = self.fe.maxpool(x)

        l1out = self.fe.layer1(x)
        l2out = self.fe.layer2(l1out)
        l3out = self.fe.layer3(l2out)
        if len(self.layers) == 4:
            l4out = self.fe.layer4(l3out)
            out = [l1out, l2out, l3out, l4out]
        elif len(self.layers) == 3:
            out = [l1out, l2out, l3out]
        else:
            raise NotImplementedError
        return out
    



    def forward_convnext(self, x):
        # 96 96 192 384 768
        # 0: stem      3-96, stride 4
        # 1: stage1   96-96
        # 2: conv    96-192, stride 2
        # 3: stage2 192-192
        # 4: conv   192-384, stride 2
        # 5: stage3 384-384
        # 6: conv   384-768, stride 2
        # 7: stage4 768-768
        layers_list = list(self.fe.features.children())
        # assert len(layers_list)==8
        # if len(self.layers) == 3:
        #     layers_list = layers_list[:-2]
        # elif len(self.layers) == 4:
        #     layers_list = layers_list
        # else:
        #     raise NotImplementedError
        out = []
        for i in range(len(layers_list)):
            layer = layers_list[i]
            if i == 1: 
                layer = layer[:self.layers[0]]
            if i == 3: 
                layer = layer[:self.layers[1]]
            if i == 5: 
                layer = layer[:self.layers[2]]
            if i == 7:
                layer = layer[:self.layers[3]]
            x = layer(x)
            if i in [1,3,5]: 
                out.append(x)
        return out
    

    def forward(self, x):
        if self.fe_type in ['resnet18', 'resnet34', 'resnet50']:
            x_list = self.forward_resnet(x)
        if self.fe_type in ['resnet18unet']:
            x_list = self.forward_resnetunet(x)
        if self.fe_type in ['darknet']:
            x_list = self.forward_darknet(x)
        if self.fe_type in ['squeezesegv2']:
            x_list = self.forward_squeezesegv2(x)
        if self.fe_type in ['squeezenet10', 'squeezenet11']:
            x_list = self.forward_squeezenet(x)
        if self.fe_type in ['convnext_tiny']:
            x_list = self.forward_convnext(x)
        if 'clip' in self.fe_type:
            x_list = self.forward_clip(x)
        if 'dinov2' in self.fe_type:
            x_list = self.forward_dino(x)

        # feature map
        feat_map = x_list[-1]

        return feat_map, x_list
        
