import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.functional as F
import argparse
import time
import numpy as np
import sys
import cv2
import copy
import subprocess
from tqdm import tqdm
import os
from torch.optim import lr_scheduler

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
########## Hyperparams ######
num_epochs = 5
lr = 0.001
########## Loader #########
def load_dataset():
	data_transforms = {
		'train': transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean,std)
		]),
		'val': transforms.Compose([
			transforms.Scale(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean,std)
		]),
		'test': transforms.Compose([
			transforms.Scale(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean,std)
		])
	}
	data_dir = 'data'
	image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x),
									data_transforms[x])
						for x in ['train', 'val', 'test']}
	data_loaders = {x : torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
												num_workers=12, shuffle=True)
						for x in ['train', 'val', 'test']}
	data_size = {x : len(image_datasets[x]) for x in ['train', 'val', 'test']}
	return data_loaders, data_size

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.model = models.resnet34(pretrained=True)
		for param in self.model.parameters():
			param.requires_grad = False
		self.model.fc = nn.Linear(self.model.fc.in_features,2,bias=False)
	def forward(self, x):
		x = self.model(x)
		return x

def train(data_loader, data_size):
	model = Model()
	model = model.cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.model.fc.parameters(), lr=lr, momentum=0.9)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	######## finetuning fc only ######
	for epoch in range(num_epochs):
		tqdm.write('Epoch {}/{}'.format(epoch, num_epochs-1))
		######### training #######
		for mode in ['train', 'val']:
			if mode == 'train':
				scheduler.step()
				model.train=True
				tot_loss = 0.0
			else:
				model.train = False
			correct = 0
			for data in tqdm(data_loader[mode]):
				inputs,labels = data
				inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
				if mode == 'train':
					optimizer.zero_grad()
				outputs = model(inputs)
				# print(outputs)
				_, preds = torch.max(outputs.data,1)
				if mode == 'train':
					loss = criterion(outputs, labels)
					loss.backward()
					optimizer.step()
					tot_loss += loss.data[0]
				correct += torch.sum(preds == labels.data)
			### Epoch info ####
			if mode == 'train':
				epoch_loss = tot_loss/data_size[mode]
				print('train loss: ', epoch_loss)
			epoch_acc = correct/data_size[mode]
			print(mode + ' acc: ', epoch_acc)
	return model
def test(data_loader, data_size, model):
	model.train = False
	correct = 0
	for data in tqdm(data_loader['test']):
		inputs, labels = data
		inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
		outputs = model(inputs)
		_, preds = torch.max(outputs.data,1)
		correct += torch.sum(preds == labels.data)
	print('test acc: ', correct/data_size['test'])
def main():
	data_loader, data_size = load_dataset()
	model =	train(data_loader, data_size)
	test(data_loader, data_size, model)
if __name__ == '__main__':
	main()	
