import os
from PIL import Image

from tqdm import tqdm, trange

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from image_helper import crop_big_image, grid_crop, display_grid_crop, get_tensor
from data_helper import CustomDataset
from model import Encoder

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomAffine(degrees=0, translate=(0.25, 0.25), scale=(0.9, 1.1), fillcolor=(0, 10, 10)),
#     torchvision.transforms.ColorJitter(brightness=0.3, contrast=0, saturation=0, hue=0),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.3190, 0.2954, 0.3324], std=[0.2802, 0.2710, 0.2786]),
    ])
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.3190, 0.2954, 0.3324], std=[0.2802, 0.2710, 0.2786]),
    ])

trainset = CustomDataset(root='./resized_data', transform=[train_transform, train_transform])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=74, shuffle=True, num_workers=4, drop_last=True)
testset = CustomDataset(root='./resized_data', transform=[train_transform, train_transform])
testloader = torch.utils.data.DataLoader(testset, batch_size=148, shuffle=False, num_workers=4)

# negset = CustomDataset(root='./neg_data', transform=[train_transform])
# negloader = torch.utils.data.DataLoader(negset, batch_size=74, shuffle=True, num_workers=4, drop_last=True)

encoder = Encoder().cuda()

classification_criterion = nn.CrossEntropyLoss()
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

pbar = tqdm(range(5001))
for epoch in pbar:
    encoder.train()
    
    running_loss = 0.0
    total = 0
    negiter = iter(negloader)
    for data in iter(trainloader):
        neg_img1, neg_img2, _ = negiter.next()
        
        img1, img2, target = data
        img1 = torch.cat([img1, neg_img1])
        img2 = torch.cat([img2, neg_img2])
        
        img1 = img1.cuda()
        img2 = img2.cuda()
        batch_size = img1.shape[0]
        iu = np.triu_indices(batch_size, k=1)
        
#         label = np.random.choice(batch_size)
        
        rep1 = encoder(img1)
        rep2 = encoder(img2)
#         negrep = encoder(neg_img)
        
        cos_matrix = torch.einsum('ik,jk->ij', [rep1, rep2])
#         cos_matrix = (cos_matrix / 0.1).exp()
#         cos_matrix2 = torch.einsum('ik,jk->ij', [rep1, negrep])
        
        cos_mean1 = cos_matrix.diag().mean()
        cos_mean2 = cos_matrix[iu].topk(batch_size)[0].mean()
        
#         cos_mean3 = cos_matrix2.topk(batch_size)[0].mean()
        
#         loss = classification_criterion(cos_matrix.unsqueeze(0), torch.LongTensor([label]).cuda())
        loss = cos_mean2 - cos_mean1
        encoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        
        running_loss += loss.item()
        total += 1
    
    with torch.no_grad():
        if epoch % 500 == 0:
            encoder.eval()

            for data in iter(testloader):
                img1, img2, target = data
                img1 = img1.cuda()
                img2 = img2.cuda()
                batch_size = img1.shape[0]

                rep1 = encoder(img1)
                rep2 = encoder(img2)

                cos_matrix = torch.einsum('ik,kj->ij', [rep1, rep2.T])

                cos_sum1 = cos_matrix.diag().mean()

                iu = np.triu_indices(148, k=1)
                cos_array2 = cos_matrix[iu]
        
    pbar.set_description(f"loss: {running_loss / total:.4}_{cos_sum1:.4}_{cos_array2.min():.4}_{cos_array2.mean():.4}_{cos_array2.max():.4}")
        
torch.save(encoder.state_dict(), 'saved_model.pth')