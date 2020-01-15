import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse

from torch.utils.data import DataLoader
from torch.autograd import Variable

## choh
from models import ETER_GRU_DFU_bnx
from models import dataloader_fastmri_brain
from config import *





# Device configuration
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)






def main(args):
	print('\n  choh, test, FastMRI brain, 384x396x16ch, @main')
	idx  = args.idx
	flag_cmap = args.cmap

	
	model = ETER_GRU_DFU_bnx()
	model = torch.load('tensors_R4.pt', map_location="cuda")
	model.eval()



	choh_dataloader_test = dataloader_fastmri_brain()
	print(' len(choh_data_test) : %d'%len(choh_dataloader_test))

	testloader = DataLoader(choh_dataloader_test, batch_size=2, shuffle=False)
	total_step = len(testloader)

	criterion = nn.MSELoss()

	inputs = []
	results = []
	refs = []

	with torch.no_grad():
		print('\n  start inferece')
		for i_batch, sample_batched in enumerate(testloader):
			data_in = sample_batched['data']
			data_in = data_in.type(torch.cuda.FloatTensor)
			# data_in = data_in.type(torch.FloatTensor)

			data_ref = sample_batched['label']
			data_ref = data_ref.type(torch.cuda.FloatTensor)
			# data_ref = data_ref.type(torch.FloatTensor)

			out = model(data_in)

			inputs = np.append( inputs, data_in.cpu().detach().numpy() )
			results = np.append( results, out.cpu().detach().numpy() )
			refs = np.append( refs, data_ref.cpu().detach().numpy() )

			loss = criterion(out, data_ref)
			print('  {} loss: {:.6f}'.format(i_batch, loss ) )
	inputs = np.reshape(inputs, [len(choh_dataloader_test), N_INPUT_VERTICAL, N_FREQ_ENCODING*N_RFCOIL*2])
	results = np.reshape(results, [len(choh_dataloader_test), N_OUT_X, N_OUT_Y])
	refs = np.reshape(refs, [len(choh_dataloader_test), N_OUT_X, N_OUT_Y])

	flag_while = True
	while flag_while:
		plt.figure()

		img_input = np.squeeze( inputs[idx,:,:])
		img_pred = np.squeeze( results[idx,:,:] )
		img_truth = np.squeeze( refs[idx,:,:] )

		plt.subplot(4,2,1)
		plt.imshow(img_pred, aspect='equal', cmap=flag_cmap)
		plt.title('img_pred')
		plt.colorbar()

		plt.subplot(4,2,2)
		plt.imshow(img_truth, aspect='equal', cmap=flag_cmap)
		plt.title('img_truth')
		plt.colorbar()

		plt.subplot(4,2,3)
		plt.imshow(img_input, aspect='equal', cmap=flag_cmap)
		plt.title('img_input')
		plt.colorbar()

		plt.subplot(4,2,4)
		plt.imshow(np.abs(img_truth-img_pred), aspect='equal', cmap=flag_cmap)
		plt.title('diff')
		plt.colorbar()
		plt.show()


		try:
			print(' ')
			x = int(input("Enter a number (idx): "))
		except ValueError:
			print('    not a number, end')
			flag_while = False
			break
		idx = x




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='    choh, get array idx for display')
	parser.add_argument('-i', '--idx', type=int, default=3, help='array idx for display')
	parser.add_argument('-c','--cmap', type=str, default='viridis', help='colormap for display')
	args = parser.parse_args()

	main(args)











	
