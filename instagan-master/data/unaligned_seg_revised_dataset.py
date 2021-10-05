import sys
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
# from jTransform import JResize, JNormalize, JToTensor, JRandomCrop, JRandomHorizontalflip
# from jTransform import *

import numpy as np

class UnalignedSegRevisedDataset(BaseDataset):
	def name(self):
		return 'UnalignedSegRevisedDataset'

	@staticmethod
	def modify_commandline_options(parser, is_train):
		return parser

	def initialize(self, opt):
		self.opt = opt
		self.root = opt.dataroot
		self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
		self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
		self.max_instances = 20  # default: 20
		self.seg_dir = 'seg'  # default: 'seg'

		self.A_paths = sorted(make_dataset(self.dir_A))
		self.B_paths = sorted(make_dataset(self.dir_B))
		self.A_size = len(self.A_paths)
		self.B_size = len(self.B_paths)
		# self.transform = get_transform(opt)

	# def fixed_transform(self, image, seed):
	# 	random.seed(seed)
	# 	return self.transform(image)

	def read_segs(self, seg_path):
		segs = list()
		for i in range(self.max_instances):
			path = seg_path.replace('.png', '_{}.png'.format(i))
			if os.path.isfile(path):
				seg = Image.open(path).convert('L')
				# seg = self.fixed_transform(seg, seed)
				segs.append(seg)
			else:
				segs.append(-torch.ones(segs[0].size()))
		return torch.cat(segs)

	def readAndTransform(self, A_path, B_path, A_seg_path, B_seg_path, prob=0.5):
		A = Image.open(A_path).convert('RGB')
		B = Image.open(B_path).convert('RGB')
		seed = random.random()
		Ai, Aj, Ah, Aw = T.RandomCrop.get_params(A, (self.opt.fineSizeH, self.opt.fineSizeW))
		Bi, Bj, Bh, Bw = T.RandomCrop.get_params(B, (self.opt.fineSizeH, self.opt.fineSizeW))

		A = TF.resize(A, (self.opt.loadSizeH, self.opt.loadSizeW), TF.InterpolationMode.BICUBIC)
		B = TF.resize(B, (self.opt.loadSizeH, self.opt.loadSizeW), TF.InterpolationMode.BICUBIC)
		A = TF.crop(A, Ai, Aj, Ah, Aw)
		B = TF.crop(B, Bi, Bj, Bh, Bw)
		if prob > seed:
			A = TF.hflip(A)
			B = TF.hflip(B)
		A = TF.to_tensor(A)
		B = TF.to_tensor(B)
		A = TF.normalize(A, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		B = TF.normalize(B, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

		A_segs = list()
		B_segs = list()
		for i in range(self.max_instances):
			A_path = A_seg_path.replace('.png', '_{}.png'.format(i))
			B_path = B_seg_path.replace('.png', '_{}.png'.format(i))
			if os.path.isfile(A_path):
				A_seg = Image.open(A_path).convert('L')
				A_seg = TF.resize(A_seg, (self.opt.loadSizeH, self.opt.loadSizeW), TF.InterpolationMode.BICUBIC)
				A_seg = TF.crop(A_seg, Ai, Aj, Aw, Ah)
				if prob >= seed:
					A_seg = TF.hflip(A_seg)
				A_seg = TF.to_tensor(A_seg)
				A_seg = TF.normalize(A_seg, (0.5), (0.5))
				A_segs.append(A_seg)

			if os.path.isfile(B_path):
				B_seg = Image.open(B_path).convert('L')
				B_seg = TF.resize(B_seg, (self.opt.loadSizeH, self.opt.loadSizeW), TF.InterpolationMode.BICUBIC)
				B_seg = TF.crop(B_seg, Bi, Bj, Bw, Bh)
				if prob >= seed:
					B_seg = TF.hflip(B_seg)
				B_seg = TF.to_tensor(B_seg)
				B_seg = TF.normalize(B_seg, (0.5), (0.5))
				B_segs.append(B_seg)


		A_segs = torch.cat(A_segs)
		B_segs = torch.cat(B_segs)
		# print(A_segs.size())
		# print(B_segs.size())

		# A, B, A_segs, B_segs = transform((A, B, A_segs, B_segs))
		A_segs = torch.cat((A_segs, -torch.ones( (20-A_segs.size(0),
							A_segs.size(1),
							A_segs.size(2)) )))

		B_segs = torch.cat((B_segs, -torch.ones( (20-B_segs.size(0),
							B_segs.size(1),
							B_segs.size(2)) )))

		# print(A_segs.size())
		# print(B_segs.size())

		return A, B, A_segs, B_segs


	def __getitem__(self, index):
		index_A = index % self.A_size
		if self.opt.serial_batches:
			index_B = index % self.B_size
		else:
			index_B = random.randint(0, self.B_size - 1)

		A_path = self.A_paths[index_A]
		B_path = self.B_paths[index_B]
		A_seg_path = A_path.replace('A', 'A_{}'.format(self.seg_dir))
		B_seg_path = B_path.replace('B', 'B_{}'.format(self.seg_dir))

		A_idx = A_path.split('/')[-1].split('.')[0]
		B_idx = B_path.split('/')[-1].split('.')[0]

		# transfome = T.Compose([JResize((220,220)),
		# 					   JRandomCrop((200, 200)),
		# 					   JRandomHorizontalflip(0.5),
		# 					   # JToTensor(),
		# 					   JNormalize()
		# 					   ])

		A, B, A_segs, B_segs = self.readAndTransform(A_path, B_path, A_seg_path, B_seg_path)

		# print('(A, B) = (%d, %d)' % (index_A, index_B))
		# seed = random.randint(-sys.maxsize, sys.maxsize)

		# A = Image.open(A_path).convert('RGB')
		# B = Image.open(B_path).convert('RGB')

		# A = self.fixed_transform(A, seed)
		# B = self.fixed_transform(B, seed)

		# A_segs = self.read_segs(A_seg_path, seed)
		# B_segs = self.read_segs(B_seg_path, seed)



		if self.opt.direction == 'BtoA':
			input_nc = self.opt.output_nc
			output_nc = self.opt.input_nc
		else:
			input_nc = self.opt.input_nc
			output_nc = self.opt.output_nc

		if input_nc == 1:  # RGB to gray
			tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
			A = tmp.unsqueeze(0)
		if output_nc == 1:  # RGB to gray
			tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
			B = tmp.unsqueeze(0)

		return {'A': A, 'B': B,
				'A_idx': A_idx, 'B_idx': B_idx,
				'A_segs': A_segs, 'B_segs': B_segs,
				'A_paths': A_path, 'B_paths': B_path}

	def __len__(self):
		return max(self.A_size, self.B_size)


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataroot',
						default='../datasets/shp2gir_coco',
						help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
	parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

	parser.add_argument('--loadSizeW', type=int, default=220, help='scale images to this size (width)')
	parser.add_argument('--loadSizeH', type=int, default=220, help='scale images to this size (height)')
	parser.add_argument('--fineSizeW', type=int, default=200, help='then crop to this size (width)')
	parser.add_argument('--fineSizeH', type=int, default=200, help='then crop to this size (height)')
	parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
						help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
	parser.add_argument('--no_flip', action='store_true',
						help='if specified, do not flip the images for data augmentation')
	parser.add_argument('--serial_batches', action='store_true', default=False,
						help='if true, takes images in order to make batches, otherwise takes them randomly')
	parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
	parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
	parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

	args = parser.parse_args(args=[])
	args.isTrain = 'train'

	dataset = UnalignedSegRevisedDataset()
	dataset.initialize(args)

	dataloader = torch.utils.data.DataLoader(
		dataset,
		batch_size=1,
		shuffle=True,
		num_workers=int(1))

	iters = iter(dataloader)
	data = iters.next()

	# for k, v in data.items():
	# 	print(k, type(v[0]))

	# print(data['A'].size())
	# print(data['B'].size())
	# print(data['A_segs'].size())
	# print(data['B_segs'].size())
	#
	print('A size', data['A'].size())
	print('B size', data['B'].size())
	print('A seg size', data['A_segs'].size())
	print('B seg size',  data['B_segs'].size())
	print()


	# for i in range(3):
	# 	print(f'A mean {i}', data['A'][:,i,:,:].mean())
	# 	print(f'B mean {i}', data['B'][:,i,:,:].mean())
	#
	# 	print(f'A min max {i}', data['A'][:,i,:,:].min(), data['A'][:,i,:,:].max())
	# 	print(f'B min max {i}', data['B'][:,i,:,:].min(), data['B'][:,i,:,:].max())
	# 	print()
	#
	# print()
	# for i in range(20):
	# 	print(f'A_segs mean {i}', data['A_segs'][:,i,:,:].mean())
	# 	print(f'B_segs mean {i}', data['B_segs'][:,i,:,:].mean())
	#
	# 	print(f'A_segs min max {i}', data['A_segs'][:,i,:,:].min(), data['A_segs'][:,i,:,:].max())
	# 	print(f'B_segs min max {i}', data['B_segs'][:,i,:,:].min(), data['B_segs'][:,i,:,:].max())
	# 	print()
	#
	# #
	#
	# from torchvision.utils import make_grid, save_image
	#
	# dataAs_grid = data['A_segs'].permute(1,0,2,3).repeat(1,3,1,1)
	# dataBs_grid = data['B_segs'].permute(1,0,2,3).repeat(1,3,1,1)
	#
	# setA = torch.cat((data['A'], dataAs_grid))
	# setB = torch.cat((data['B'], dataBs_grid))
	# setAB = torch.cat((setA, setB))
	# save_image((setAB + 1) / 2, './revised_save_img.png', nrow=21)




	# dataA_grid = make_grid(data['A'])
	# dataB_grid = make_grid(data['B'])
	# dataAs_grid = make_grid(data['A_segs'].permute(1,0,2,3), nrow=1)
	# dataBs_grid = make_grid(data['B_segs'].permute(1,0,2,3), nrow=1)
	#
	# print(dataAs_grid.size())
	# data = torch.cat([dataA_grid, dataAs_grid], dim=)


	# import matplotlib.pyplot as plt
	# plt.imshow((data + 1 / 2).permute(1, 2, 0))
	# plt.show()
	#
	# print(dataA_grid.size(), dataB_grid.size(), dataAs_grid.size(), dataBs_grid.size())

	# plt.imshow((dataAs_grid + 1 / 2).permute(1,2,0))
	# plt.show()
	# plt.imshow((dataBs_grid + 1 / 2).permute(1,2,0))
	# plt.show()


	# t = data['A_segs']
	# t += 1
	# t = t.mean(0).mean(-1).mean(-1)
	# print(t.size())
	# print(t)
	# print('min', t.min(), 'max', t.max())
	# print(t)

	# li = list()
	# for i in range(t.size(1)):
	# 	seg = t[:, i, :, :].unsqueeze(1)
	# 	li.append(seg)
	#
	# li = torch.cat(li)
	# print(li.size())
	# print(torch.sum(li, dim=0, keepdim=True).size())

	# dataset[1]
	# data = dataset[0]

	# seg = data['A_segs']
	# print(seg.size())
	# print(seg.mean(-1))
	# print(seg)



	# print(data['A_idx'])
	# print(data['B_idx'])
	#

	# for segs in data['A_segs']:
	# 	print(segs.size())
	# 	print(segs.min(), segs.max())
	# 	mean = segs.mean(-1).mean(-1)
	# 	print(mean)
	# 	m, i = mean.topk(4)
	# 	print(m, i)
		# ret.append(segs[i, :, :])

	# for i in range(20):
	# 	target = data['A_segs'].permute(1,2,0)[...,i]
	# 	print(target.min(), target.max())
	# 	plt.imshow((target + 1 / 2))
	# 	plt.show()

	# plt.imshow((data['B_segs'] + 1 / 2).permute(1,2,0))
	# plt.show()

	#
	# plt.imshow((data['A_segs'] + 1 / 2).permute(1, 2, 0))
	# plt.show()
	# plt.imshow((data['B_segs'] + 1 / 2).permute(1, 2, 0))
	# plt.show()