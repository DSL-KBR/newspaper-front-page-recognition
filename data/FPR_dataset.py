import cv2 as cv
from PIL import Image, UnidentifiedImageError

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from data.base_dataset import FPRDataRegister, sample_query

# images are resized to fit the following dimensions
imgH = 1024
imgW = 1024


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

    def __init__(self, DataRegister, mode='train'):
        """
        -- DataRegister: register all images for FPR training and testing
        -- mode: train or test
        """

        assert isinstance(DataRegister, FPRDataRegister), 'FPRDataRegister must be applied first ' \
                                                          'before a FPR datase can be created'

        imgmetadata = DataRegister.imgmetadata
        datasetdir = DataRegister.imgdir
        self.S_transform = get_transform(resize=[imgH, imgW])

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

        # completely paranoid coding below
        label = self.label[index]
        metadata = self.metadata[index]
        if label == 0:
            assert metadata['Page'] != 1, f"{self.metadata[index]} is wrongly registered as a non front page"
        if label == 1:
            assert metadata['Page'] == 1, f"{self.metadata[index]} is wrongly registered as a non front page"

        return {'Sample': S, 'Label': label, 'Path': S_path, 'Metadata': metadata}

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
