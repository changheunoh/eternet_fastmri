# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F




class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, depth=5, wf=6, padding=True, batch_norm=False, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        #
        Using the default arguments will yield the exact version used
        in the original paper
        #
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)
            #
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)
            #
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
        #
    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)
                #
        for i, up in enumerate(self.up_path):
            # print(' i : %d'%i)
            # print(x.shape)
            # print(blocks[-i-1].shape)

            if i is 0:
                x = up(x, blocks[-i-1])
            else:
                x = up(x, blocks[-i-1], blocks[-i])
            ###choh, dual frame unet
            # x = up(x-blocks[-i-1], blocks[-i-1]) ## wrong
            #
        last_x = self.last(x)
        return last_x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []
        #
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
            #
        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
            #
        self.block = nn.Sequential(*block)
        #
    def forward(self, x):
        out = self.block(x)   ### concat [x, out ]
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))
            #
        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)
        #
    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]
        #
    def forward(self, x, bridge, residual=None):
        # print(' i : %d'%i)
        # print(x.shape)    ### concat [x, bridge[-1] ]  , (should concat before up(x))
        # print(bridge.shape)
        if residual is not None:
            # print('residual')
            # print(residual.shape)
            x = torch.sub(x, residual)

        up = self.up(x)
        # print(up.shape)
        # print(up.shape[2:])
        crop1 = self.center_crop(bridge, up.shape[2:])
        # print(crop1.shape)
        out = torch.cat([up, crop1], 1)
        # print(out.shape)
        out = self.conv_block(out)
        # print(out.shape)
        #
        return out




class UNet_choh_skip(nn.Module):
	def __init__(self, in_channels=1, n_classes=1, depth=5, wf=6, padding=True, batch_norm=False, up_mode='upconv', n_hidden=16):
		super(UNet_choh_skip, self).__init__()
		assert up_mode in ('upconv', 'upsample')
		self.padding = padding
		self.depth = depth
		prev_channels = in_channels
		self.down_path = nn.ModuleList()
		for i in range(depth):
		    self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i), padding, batch_norm))
		    prev_channels = 2**(wf+i)
		self.up_path = nn.ModuleList()
		for i in reversed(range(depth - 1)):
		    self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode, padding, batch_norm))
		    prev_channels = 2**(wf+i)
		self.last = nn.Conv2d(prev_channels+n_hidden*2, n_classes, kernel_size=1)
	def forward(self, x):
		blocks = []
		blocks.append(x)
		for i, down in enumerate(self.down_path):
		    x = down(x)
		    if i != len(self.down_path)-1:
		        blocks.append(x)
		        x = F.avg_pool2d(x, 2)
		for i, up in enumerate(self.up_path):
		    if i is 0:
		        x = up(x, blocks[-i-1])
		    else:
		        x = up(x, blocks[-i-1], blocks[-i])
		out = torch.cat([blocks[0], x], 1)
		last_x = self.last(out)
		return last_x




def main():
    print('\n  choh, UNet testing . . . ')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = UNet().to(device)
    print(model)






if __name__ == '__main__':
    main()
