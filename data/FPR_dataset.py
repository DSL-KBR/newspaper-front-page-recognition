""" Dataset and DataLoader
Load data from a single folder
Return:
    { A, A_paths, A_label}
This dataset will be used when testing the network
"""
import cv2 as cv
from PIL import Image, UnidentifiedImageError
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from data.base_dataset import FPRDataRegister, sample_query

# images are resized to fit the below dimensions
imgH = 1024
imgW = 1024
gridX = torch.arange(start=0, step=imgW/4, end=imgW+1).long()
gridY = torch.arange(start=0, step=imgH/4, end=imgH+1).long()


class FPRDataLoader:

    def __init__(self, dataset, batchsize=1, num_threads=4, isshuffle=True):
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batchsize,
            shuffle=isshuffle,
            num_workers=int(num_threads),
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        """ return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data


class FPRDataset(data.Dataset):
    """Loading images
    -- <__init__>:                      initialize the class
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    """

    def __init__(self, DataRegister, mode='train'):
        """
        -- DataRegister: register all images for FPR training and testing
        -- mode: train or test
        """

        assert isinstance(DataRegister, FPRDataRegister), 'FPRDataRegister is missing'

        imgmetadata = DataRegister.imgmetadata
        datasetdir = DataRegister.imgdir
        self.S_transform = get_transform(resize=[imgH, imgW])

        if hasattr(DataRegister, 'attack_dir'):
            self.attack_dir = DataRegister.attack_dir
            self.attack_transform = transforms.Compose([
                transforms.Resize([256,256], Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )])
            self.attack_location = DataRegister.attack_location

        if mode == 'train':
            traintestidx = DataRegister.trainidx
        elif mode == 'test':
            traintestidx = DataRegister.testidx

        self.paths, self.label, self.metadata = make_dataset(traintestidx, datasetdir, imgmetadata)

    def __getitem__(self, index):

        # load sample image
        S_path = self.paths[index]
        try:
            S = Image.open(S_path)
        except UnidentifiedImageError:
            S = Image.fromarray(cv.imread(S_path))
        S = torch.squeeze(self.S_transform(S))
        if len(S.shape) == 2:
            S = torch.stack([S, S, S], 0)

        label = self.label[index]
        metadata = self.metadata[index]
        if label == 0:
            assert metadata['Page'] != 1, f"{self.metadata[index]} is wrongly registered as a non front page"
        if label == 1:
            assert metadata['Page'] == 1, f"{self.metadata[index]} is wrongly registered as a non front page"

        # creating S_augment by attacking S using S_attack
        S_augment = S.detach().clone()
        attack_location_sampling = []
        if hasattr(self, 'attack_dir'):
            attack_location_sampling = self.attack_location.copy()
            num_add_samples = 5 - len(attack_location_sampling)
            while num_add_samples > 0:
                attack_location_sampling.extend(random.sample(range(0, 16), num_add_samples))
                attack_location_sampling = set(attack_location_sampling)
                num_add_samples = 5 - len(attack_location_sampling)

            attack_location_sampling = torch.as_tensor(list(attack_location_sampling))
            for attack_location in attack_location_sampling:
                # first pick an attacking image
                attack_path = self.attack_dir[random.sample(range(0, len(self.attack_dir)), 1)[0]]
                S_attack = self.attack_transform(Image.open(attack_path))
                if S_attack.shape[0] == 1 or len(S_attack) == 2:
                    print('error')
                # attack sample image at attack_location
                location_y = torch.div(attack_location, 4, rounding_mode='floor')
                location_x = attack_location % 4
                S_augment[:, gridY[location_y]:gridY[location_y + 1], gridX[location_x]:gridX[location_x + 1]] = S_attack

        return {'Sample': S, 'Label': label, 'Path': S_path, 'Metadata': metadata,
                'Sample_augment': S_augment, 'Sample_augment_location': attack_location_sampling}

    def __len__(self):
        return len(self.label)


def get_transform(resize=[0, 0]):
    transform_list = []
    if resize == [0, 0]:
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize([0.5], [0.5])]
    else:
        transform_list += [transforms.Resize(resize)]
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize([0.5], [0.5])]

    return transforms.Compose(transform_list)


def make_dataset(traintestidx, datasetdir, samplemetadata):
    imgdir = []
    imglabel = []
    imgmetadata = []
    for i in traintestidx:
        dir_sample = datasetdir[i]
        metadata_sample = sample_query(dir_sample)
        if metadata_sample == samplemetadata[i]:
            imgdir.append(dir_sample)
            imgmetadata.append(metadata_sample)
            if metadata_sample['Page'] == 1:
                imglabel.append(1)  # front pages are labeled as 1s
            else:
                imglabel.append(0)  # non-front pages are labeled as 0s

    return imgdir, imglabel, imgmetadata
