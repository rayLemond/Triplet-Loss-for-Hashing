import os, argparse, torch, time
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as torchdata
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import itertools

####################################################### utils ##################################################
def default_loader(path):
	return Image.open(path).convert('RGB')

class MyDataset(torchdata.Dataset):
	def __init__(self, path, txt, transform=None, loader=default_loader):
		self.transform = transform
		self.path = path
		self.loader = loader
		fh = open(txt, 'r')
		imgs = []
		for line in fh:
			if line == '':
				break
			line = line.strip('\n')
			line = line.rstrip()
			words = line.split(' ')
			imgs.append(words)
		self.imgs = imgs

	def __getitem__(self, index):
		words = self.imgs[index]
		img = self.loader(self.path + words[0])  # path of your data files
		img = self.transform(img)
		label = int(words[1])
		return img, label

	def __len__(self):
		return len(self.imgs)

def myDataloader(opt):
	batchSize = opt.batchSize
	normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
	# normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
	transform_train = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		normalize
	])
	transform_test = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		normalize
	])
	if opt.dataset == 'CUB':
		train_dir = opt.imgPath + 'CUB_Train.txt'
		test_dir = opt.imgPath + 'CUB_Test.txt'
		DB_dir = opt.imgPath + 'CUB_DB.txt'
		imgPath = opt.imgPath + 'CUB_200_2011/images/'

	test_set = MyDataset(imgPath, txt=test_dir, transform=transform_test)
	test_loader = torchdata.DataLoader(test_set, batch_size=batchSize,shuffle=False,
									   num_workers=2, pin_memory=True)
	train_set = MyDataset(imgPath, txt=train_dir, transform=transform_train)
	train_loader = torchdata.DataLoader(train_set, batch_size=batchSize, shuffle=True,
									   num_workers=2, pin_memory=True)
	DB_set = MyDataset(imgPath, txt=DB_dir, transform=transform_test)
	DB_loader = torchdata.DataLoader(DB_set, batch_size=batchSize, shuffle=False,
									   num_workers=2, pin_memory=True)

	return DB_loader, train_loader, test_loader


def combinations(iterable, r):
	pool = list(iterable)
	n = len(pool)
	for indices in itertools.permutations(range(n), r):
		if sorted(indices) == list(indices):
			yield list(pool[i] for i in indices)


def get_triplets(labels):
	labels = labels.cpu().data.numpy()
	triplets = []
	for label in set(labels):
		label_mask = (labels == label)
		label_indices = np.where(label_mask)[0]
		if len(label_indices) < 2:
			continue
		negative_indices = np.where(np.logical_not(label_mask))[0]
		anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

		# Add all negatives for all positive pairs
		temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
						 for neg_ind in negative_indices]
		triplets += temp_triplets
	return torch.LongTensor(np.array(triplets))


def compute_result(dataloader, net, opt):
	bs, clses = [], []
	net.eval()
	with torch.no_grad():
		for img, cls in dataloader:
			clses.append(cls)
			img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
			bs.append(net(img.cuda()))
	return torch.sign(torch.cat(bs)), torch.cat(clses)

def compute_mAP(trn_binary, tst_binary, trn_label, tst_label):
	for x in trn_binary, tst_binary, trn_label, tst_label: x.long()

	AP = []
	Ns = torch.arange(1, trn_binary.size(0) + 1)
	Ns = Ns.type(torch.FloatTensor)
	for i in range(tst_binary.size(0)):
		query_label, query_binary = tst_label[i], tst_binary[i]
		_, query_result = torch.sum((query_binary != trn_binary).long(), dim=1).sort()
		correct = (query_label == trn_label[query_result]).float()
		P = torch.cumsum(correct.type(torch.FloatTensor), dim=0) / Ns
		AP.append(torch.sum(P * correct) / torch.sum(correct))
	mAP = torch.mean(torch.Tensor(AP))
	return mAP

def choose_gpu(i_gpu):
	"""choose current CUDA device"""
	torch.cuda.device(i_gpu).__enter__()
	cudnn.benchmark = True

def feed_random_seed(seed=np.random.randint(1, 10000)):
	"""feed random seed"""
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

def saveHashCodes(trn_binary, tst_binary, trn_label, tst_label, mAP, opt):
	saved = dict()
	saved['db_hcodes'] = trn_binary
	saved['test_hcodes'] = tst_binary
	saved['db_labels'] = trn_label
	saved['test_labels'] = tst_label
	torch.save(saved, opt.savePath + f'TripletH_{opt.dataset}_{opt.bits}bits_[{mAP*100:.2f}]mAP.pkl')

####################################################### main ##################################################

def triplet_hashing_loss_regu(embeddings, triplets, margin):

	pos_dist = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
	neg_dist = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
	losses = F.relu(pos_dist - neg_dist + margin)	# hinge loss

	return losses.mean()

def train(dataloader, net, optimizer, opt):
	sum_lossT = 0
	sum_lossQ = 0
	net.train()

	for i, (img, cls) in enumerate(dataloader):	

		net.zero_grad()
		embeddings = net(img.cuda())
		embeddings = torch.tanh(embeddings)
		triplets = get_triplets(cls)	

		if triplets.shape[0] != 0:
			lossT = triplet_hashing_loss_regu(embeddings, triplets, opt.margin)
			lossQ = torch.abs(torch.pow(embeddings.abs() - torch.ones(embeddings.size()).cuda(), 3)).mean()
			loss = lossT + opt.lambdaQ * lossQ	# triplet loss + lambdaQ * quantization loss
			loss.backward()		
			optimizer.step()
			sum_lossT += float(lossT)
			sum_lossQ += float(lossQ)
		else:
			del embeddings

	return sum_lossT/len(dataloader), sum_lossQ/len(dataloader)

def main():
	parser = argparse.ArgumentParser(description='Triplet Loss')
	parser.add_argument('--checkpoint', type=int, default=20, help='check mAP')
	parser.add_argument('--obpoint', type=int, default=10, help='show loss')
	parser.add_argument('--batchSize', type=int, default=100)
	parser.add_argument('--ngpu', type=int, default=0, help='gpu id')	
	parser.add_argument('--bits', type=int, default=64, help='hash bit length')
	parser.add_argument('--niter', type=int, default=2101, help='epoch')
	parser.add_argument('--dataset', type=str, default='CUB')
	parser.add_argument('--margin', type=float, default=0, help='0.5 bits by default')
	parser.add_argument('--lambdaQ', type=float, default=0.05, help='weight para for quantization loss')
	parser.add_argument('--savePath', type=str, default='/data1/liangyuchen/Saved/HashCodes/')
	parser.add_argument('--imgPath', type=str, default='/data1/liangyuchen/Data/CUB/')
	opt = parser.parse_args()
	opt.margin = 0.5 * opt.bits
	print(opt)
	choose_gpu(opt.ngpu)
	feed_random_seed()

	DB_loader, train_loader, test_loader = myDataloader(opt)
	net = models.resnet50(pretrained=True)
	net.fc = nn.Linear(2048, opt.bits)
	net.cuda()

	optimizer = torch.optim.Adam(net.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=10**-5)
	# optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-4, weight_decay=1e-5)

	print('Start training Triplet Loss Hashing. Training Loss is shown as [epoch]: Triplet Loss + Quantization Loss.')
	max_map = 0
	for epoch in range(0, opt.niter):

		lossT, lossQ = train(train_loader, net, optimizer, opt)

		if epoch%opt.obpoint==0:
			print(f'[{epoch:4d}]: {lossT:.4f} + {lossQ:.4f}', end=' ')

		if epoch%opt.checkpoint==0 and epoch!=0:
			trn_binary, trn_label = compute_result(DB_loader, net, opt)
			tst_binary, tst_label = compute_result(test_loader, net, opt)
			mAP = compute_mAP(trn_binary, tst_binary, trn_label, tst_label)

			# if mAP>max_map and epoch>500:
				# saveHashCodes(trn_binary, tst_binary, trn_label, tst_label, mAP, opt)
				# torch.save(net.state_dict(), opt.savePath + f'TripletH_{opt.dataset}_{opt.bits}bits_[{mAP*100:.2f}]mAP_{epoch}epoch.pkl')
			
			max_map = max(mAP, max_map)
			# print(f'[{epoch:4d}] T:{lossT:.4f} Q:{lossQ:.4f}', end=' ')
			print(f'          Triplet Hashing on {opt.dataset}, [{epoch}] epoch, mAP: {mAP*100:.2f} max_mAP: {max_map*100:.2f}') 

	return max_map, opt

if __name__ == '__main__':
	max_map, opt = main()
	print(opt)
	print('[Final] retrieval mAP: ', max_map)

