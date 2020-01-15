# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import scipy.io as sio

from config import *
from myUNet_DF import UNet_choh_skip





# ETER, fastmri multi, 384x396x16ch
class ETER_GRU_DFU_bnx(nn.Module):
	def __init__(self):
		super(ETER_GRU_DFU_bnx, self).__init__()
		num_in_x = N_INPUT_VERTICAL
		num_in_y = N_FREQ_ENCODING
		n_coil = N_RFCOIL
		num_out_x = N_OUT_X
		num_out_y = N_OUT_Y
		input_size = num_in_y*n_coil*2
		num_layers = 1
		num_out1 = num_out_y*N_HIDDEN_LRNN_1
		num_in2 = num_in_x*N_HIDDEN_LRNN_1
		num_out2 = num_out_x*N_HIDDEN_LRNN_2
		num_feat_ch = int(num_out2*2/num_out_x)

		self.num_in_x = num_in_x
		self.num_in_y = num_in_y
		self.num_layers = num_layers
		self.num_out1 = num_out1
		self.num_out2 = num_out2
		self.num_out_x = num_out_x
		self.num_out_y = num_out_y

		self.gru_h = nn.GRU(input_size, num_out1, num_layers, batch_first=True, bidirectional=True)
		self.gru_v = nn.GRU(num_in2*2, num_out2, num_layers, batch_first=True, bidirectional=True)
		self.unet = UNet_choh_skip(in_channels=num_feat_ch, n_classes=1, depth=N_UNET_DEPTH, wf=6, batch_norm=False, up_mode='upconv', n_hidden=N_HIDDEN_LRNN_2)

	def forward(self, x):
		h_h0 = torch.zeros(self.num_layers*2, x.size(0), self.num_out1).cuda()
		h_v0 = torch.zeros(self.num_layers*2, x.size(0), self.num_out2).cuda()

		in_h = x.reshape([x.size(0), self.num_in_x, -1])

		out_h, _ = self.gru_h(in_h, h_h0)
		out_h = out_h.reshape([x.size(0), self.num_in_x, self.num_out_y,-1])
		out_h = out_h.permute(0, 2, 1, 3)
		out_h = out_h.reshape([x.size(0), self.num_out_y, -1])

		out_v, _ = self.gru_v(out_h, h_v0)
		out_v = out_v.reshape([x.size(0), self.num_out_y, self.num_out_x,-1])
		out_v = out_v.permute(0, 3, 2, 1)

		## merge multi feature
		out = self.unet(out_v)
		return out



class dataloader_fastmri_brain(Dataset):
	def __init__(self, transform=None):
		print('\n  Dataset : fastmri brain multi R4')
		n_freq = N_FREQ_ENCODING
		n_dim = N_INPUT_VERTICAL
		n_coilch = 16
		num_element = 10
		num_set = 2
		value_amplification = 1e4
		value_amp_X = 1e1

		path_matfolder = './'

		X_train = np.empty((num_element*num_set, n_dim, n_freq*n_coilch*2))
		Y_train = np.empty((num_element*num_set, 1, N_OUT_X, N_OUT_Y))

		## sample 1 in training set
		path_filename = 'file_brain_AXT2_200_2000003_label.mat'
		f_matfile = sio.loadmat(path_matfolder+path_filename)
		f_imgNET = f_matfile['img_all']
		YY = f_imgNET

		path_filename = 'file_brain_AXT2_200_2000003_kspace.mat'
		f_matfile_rad = sio.loadmat(path_matfolder+path_filename)
		f_imgNET_ksp = f_matfile_rad['ksp_all_16ch']
		XX = f_imgNET_ksp

		X_temp = np.empty((num_element, n_dim, n_freq, n_coilch, 2))
		X_temp[:,:,:99,:,:] = XX[:,:,3:396:4,:,:]
		X_temp[:,:,99:131,:,:] = XX[:,:,197-16:197+16,:,:]

		XX = np.reshape(X_temp, (num_element,n_dim, n_freq*n_coilch*2))
		YY = np.reshape(YY[:num_element,:,:], (num_element,1, N_OUT_X, N_OUT_Y))

		X_train[:num_element,:,:] = value_amp_X*XX
		Y_train[:num_element,:,:,:] = value_amplification*YY

		## sample 2 in test set
		path_filename = 'file_brain_AXT2_204_2040073_label.mat'
		f_matfile = sio.loadmat(path_matfolder+path_filename)
		f_imgNET = f_matfile['img_all']
		YY = f_imgNET

		path_filename = 'file_brain_AXT2_204_2040073_kspace.mat'
		f_matfile_rad = sio.loadmat(path_matfolder+path_filename)
		f_imgNET_ksp = f_matfile_rad['ksp_all_16ch']
		XX = f_imgNET_ksp

		X_temp = np.empty((num_element, n_dim, n_freq, n_coilch, 2))
		X_temp[:,:,:99,:,:] = XX[:,:,3:396:4,:,:]
		X_temp[:,:,99:131,:,:] = XX[:,:,197-16:197+16,:,:]

		XX = np.reshape(X_temp, (num_element,n_dim, n_freq*n_coilch*2))
		YY = np.reshape(YY[:num_element,:,:], (num_element,1, N_OUT_X, N_OUT_Y))

		X_train[num_element:,:,:] = value_amp_X*XX
		Y_train[num_element:,:,:,:] = value_amplification*YY

		self.label = Y_train
		self.data = X_train
		self.transform = transform

	def __len__(self):
		return self.label.shape[0]

	def __getitem__(self, idx):
		sample = {'data': self.data[idx], 'label': self.label[idx]}
		if self.transform:
			sample = self.transform(sample)
		return sample

























