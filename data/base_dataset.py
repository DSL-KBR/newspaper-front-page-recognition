""" Dataset and DataLoader
Load data from a single folder
Return:
    { A, A_paths, A_label}
This dataset will be used when testing the network
"""

import torch
import torch.utils.data as data

from PIL import Image
import os.path
import re
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]
SPLIT_METHODS = ['i', 'y', 'm', 'd']


def sample_query(sample):
    """
    This function retrives metadata information from an image sample
    """

    samplemetadata = {'IDN': -1, 'Year': -1, 'Month': -1, 'Date': -1, 'Page': -1}
    marker_ind = 'BE-KBR00'

    if isinstance(sample, str):
        info = sample
    else:
        info = sample['Path'][0]
        if isinstance(info, list):
            info = info[0]
        elif isinstance(info, tuple):
            info = info[0]

    metadatapos = info.find(marker_ind)
    if metadatapos == -1:
        print(info)
        raise Exception('unrecognized sample')
    else:
        if metadatapos != 0:
            info = info[metadatapos:]
        marker = [k.start() for k in re.finditer('_', info)]

    samplemetadata['IDN'] = int(info[marker[0] + 1:marker[1]])

    sampledate = info[marker[1] + 1:marker[2]]
    if len(sampledate) == 8:
        samplemetadata['Year'] = int(sampledate[0:4])
        samplemetadata['Month'] = int(sampledate[4:6])
        samplemetadata['Date'] = int(sampledate[6:8])
    else:
        print(info+':'+sampledate)
        raise Exception('unrecognized sample date')

    samplepage = info[marker[7] + 1:marker[8]]
    if len(samplepage) == 4:
        samplemetadata['Page'] = int(samplepage)
    else:
        print(info+':'+samplepage)
        raise Exception('unrecognized sample page')

    return samplemetadata


class FPRDataRegister:

    def __init__(self, datadir, splitindicator='m', splitseed=-1, test_ratio=0.2,
                 attack_dir='', attack_location=[]):
        """
        -- datadir: point to a directory
        -- splitindicator: train and test is splited on
                     'i': IDN of the collections
                     'y' year of the collections
                     'm' month of the collections
                     'd' date of the collections
        -- splitseed: set seed for number random generator
        -- attack_dir: dirction to attacking images
        """
        assert os.path.isdir(datadir), '%s is not a valid directory' % datadir
        self.datadir = datadir

        self.splitindicator = splitindicator.lower()
        assert self.splitindicator in SPLIT_METHODS

        if splitseed != -1:
            np.random.seed(splitseed)

        self.test_ratio = test_ratio
        if os.path.isdir(attack_dir):
            self.attack_dir = attack_dir
            self.attack_location = attack_location
            # # specific locations for attacking
            # attack_location.extend(random.sample(range(0, 16), 5))
            # self.attack_location = set(attack_location)

        self.map_dataset()

    def map_dataset(self):

        imgdir = []  # paths to images
        imgmetadata = []  # metadata of the image sample
        traintestsummary = []

        IDN_temp = []
        Year_temp = []
        Mon_temp = []
        Dat_temp = []
        Page_temp = []

        for root, directions, fnames in sorted(os.walk(self.datadir)):
            if fnames.__len__() == 0:
                continue
            else:
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        imgdir.append(path)

                        samplemetadata_temp = sample_query(fname)
                        imgmetadata.append(samplemetadata_temp)
                        IDN_temp.append(samplemetadata_temp['IDN'])
                        Year_temp.append(samplemetadata_temp['Year'])
                        Mon_temp.append(samplemetadata_temp['Month'])
                        Dat_temp.append(samplemetadata_temp['Date'])
                        Page_temp.append(samplemetadata_temp['Page'])

        self.imgdir = imgdir
        self.imgmetadata = imgmetadata
        self.SampleNumber = len(imgdir)
        traintestlabel = np.zeros(self.SampleNumber)  # train test split of images

        imgIDN = np.array([m['IDN'] for m in imgmetadata])
        # imgYear = np.array([m['Year'] for m in imgmetadata])
        imgMonth = np.array([m['Month'] for m in imgmetadata])
        # imgDate = np.array([m['Date'] for m in imgmetadata])
        imgPage = np.array([m['Page'] for m in imgmetadata])
        self.titleIDN = np.unique(imgIDN).tolist()

        if self.splitindicator == 'i':
            # doctitleIDN = self.titleIDN    # which titles are present
            # doctitleTestIndex = np.random.permutation(len(doctitleIDN))
            # doctitleTestIndex = doctitleTestIndex[0:np.max[1, np.floor(0.2 * len(doctitleTestIndex))]]
            # doctitleTest = doctitleIDN[doctitleTestIndex]
            # traintestlabel[np.isin(imgIDN, doctitleTest)] = 1
            raise NotImplementedError('spliting based on titles is not implemented')
        elif self.splitindicator == 'y':
            raise NotImplementedError('spliting based on year is not implemented')
        elif self.splitindicator == 'm':
            for doctitle in self.titleIDN:
                doctitleMon = np.unique(imgMonth[imgIDN == doctitle])   # which months are present for this doctitle
                doctitleMonTestIndex = np.random.permutation(len(doctitleMon))
                doctitleMonTestIndex = doctitleMonTestIndex[0:np.max([1, np.floor(self.test_ratio*len(doctitleMonTestIndex)).astype(int)])]

                index_Test = doctitleMon[doctitleMonTestIndex]
                index_Train = doctitleMon[~(np.isin(doctitleMon, index_Test))]
                assert ~any(np.isin(index_Test, index_Train)), f"leaking of {index_Test} into {index_Train}"

                traintestlabel[(imgIDN == doctitle) & (np.isin(imgMonth, index_Test))] = 1
                traintestsummary.append({
                                         'IDN': doctitle,
                                         'Train': {'Month': list(index_Train),
                                                   'FrontPages': sum((imgIDN == doctitle) & (imgPage == 1) & (traintestlabel == 0)),
                                                   'TotalPages': sum((imgIDN == doctitle) & (traintestlabel == 0)),
                                                   'TrainRatio': np.round(sum((imgIDN == doctitle) & (traintestlabel == 0))/sum(imgIDN == doctitle), decimals=3)},
                                         'Test': {'Month': list(index_Test),
                                                  'FrontPages': sum((imgIDN == doctitle) & (imgPage == 1) & (traintestlabel == 1)),
                                                  'TotalPages': sum((imgIDN == doctitle) & (traintestlabel == 1)),
                                                  'TestRatio': np.round(sum((imgIDN == doctitle) & (traintestlabel == 1))/sum(imgIDN == doctitle), decimals=3)},
                                         })
        elif self.splitindicator == 'd':
            raise NotImplementedError('spliting based on date is not implemented')

        self.trainidx = np.squeeze(np.where(traintestlabel == 0))
        self.testidx = np.squeeze(np.where(traintestlabel == 1))
        traintestsummary.append({'OverallTrainRatio': len(self.trainidx) / (len(self.trainidx) + len(self.testidx))})
        self.traintestmetadata = traintestsummary

        # compile attacking images
        error = []
        if hasattr(self, 'attack_dir'):
            attack_dir = []
            for root, directions, fnames in sorted(os.walk(self.attack_dir)):
                if fnames.__len__() == 0:
                    continue
                else:
                    for fname in fnames:
                        if is_image_file(fname):
                            path = os.path.join(root, fname)
                            img = Image.open(path)
                            if img.mode == 'RGB':
                                attack_dir.append(path)

            self.attack_dir = attack_dir


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
