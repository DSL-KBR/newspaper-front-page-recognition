"""
For classification of images of newspaper pages into
Non Front Page : model outputs 0
Front Page : model outputs 1

Backbone model: ResNeSt https://github.com/zhanghang1989/ResNeSt
Code developed on top of the framework published by cycleGAN https://github.com/junyanz/CycleGAN

Written by the KBR Data Science Lab https://www.kbr.be/en/projects/data-science-lab/
Published at https://github.com/DSL-KBR/newspaper-front-page-recognition
"""

import os
import time
import datetime
import argparse
import numpy as np

import torch
from data.base_dataset import FPRDataRegister
from data.FPR_dataset import FPRDataset, FPRDataLoader
from models.FPR_ResNeSt import FPRNet

from utils.utils import Logger
from utils.train_test import evaluation_metric

parser = argparse.ArgumentParser(description='Font Page Recognition (FPR) of historical newspapers')

parser.add_argument('--model_name', default='FPR_ResNeSt', type=str, help='name of the model')
parser.add_argument('--dir_data', default=[], type=str,
                    help='dir where to retrieve train and test data')
parser.add_argument('--dir_checkpoints', default=r'./checkpoints', type=str, help='dir for logging')

parser.add_argument('--num_class', default=2, type=int,
                    help='number of classes, default is two: front page/non front page')
parser.add_argument('--batch_size', default=2, type=int, help='batch size for training')
parser.add_argument('--niter', default=5, type=int,
                    help='number of epochs with constant learning rate in linear annealing')
parser.add_argument('--niter_decay', default=5, type=int,
                    help='number of epochs with reducing learning rate '
                         'in linear annealing')
parser.add_argument('--print_freq', default=1, type=int,
                    help='number of batches, set this parameter to control how '
                         'frequent training information will be eoched')
parser.add_argument('--save_epoch_freq', default=1, type=int,
                    help='number of epochs, set this parameter to control how '
                         'frequent intermediate models will be saved')

parser.add_argument('--test_ratio', default=1, type=float,
                    help='percentage of dataset for testing, if set to 1, test_epoch must be provided')
parser.add_argument('--test_epoch', default=10, type=int,
                    help='loading of a previously trained network by epoch number')

parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--train_device', default='gpu', type=str, help='cpu or gpu')

args = parser.parse_args()

if args.train_device == 'gpu':  # by default the GPU 0 is used if available
    gpu_ids = [0]
else:
    gpu_ids = []

if __name__ == '__main__':
    if args.test_ratio == 1:
        assert args.test_epoch > 0, 'For pure testing, provide a valid epoch number to load a previously trained model'

    # forcing torch to release cached GPU memory
    if args.train_device == 'gpu':
        torch.cuda.empty_cache()

    # create dir for logging
    log_dir_training = os.path.join(args.dir_checkpoints, 'training')
    log_dir_testing = os.path.join(args.dir_checkpoints, 'testing')

    for train_seed in torch.arange(start=1, step=1,
                                   end=2):  # by default, the model will be evaluated on one train-test split
        dir_model_seed = os.path.join(args.model_name, str(train_seed.item()))

        # initialize counters for iterations and epochs
        total_iters = 0
        epoch_count = 1

        """
        Register data: images hosted in dir_data will be registered and splitted into trainining and testing based on test_ratio.
        # Currently only splitting by Month within one year is supported, for other splitting configuration edit data/base_dataset 
        
        !!! The name of the images must follow a specific format, see base_dataset/sample_query for more infomration
        """
        DataRegister = FPRDataRegister(datadir=args.dir_data, splitseed=train_seed, test_ratio=args.test_ratio)

        # create train dataloader
        if args.test_ratio < 1:
            FPRTrain = FPRDataset(DataRegister=DataRegister, mode='train')
            dataloaderTrain = FPRDataLoader(dataset=FPRTrain, batchsize=args.batch_size, num_threads=args.num_workers)
            print('Training', ': the number of training images = %d' % dataloaderTrain.__len__())

        # create test dataloader
        if args.test_ratio > 0:
            FPRTest = FPRDataset(DataRegister=DataRegister, mode='test')
            dataloaderTest = FPRDataLoader(dataset=FPRTest, batchsize=10, num_threads=1)
            print('test', ': the number of testing images = %d' % (dataloaderTest.__len__()))

        # create model
        model = FPRNet(checkpoints_dir=log_dir_training, model_name=dir_model_seed, num_class=args.num_class,
                       gpu_ids=gpu_ids)

        # logging general information
        log_metadata = os.path.abspath(os.path.join(args.dir_checkpoints,
                                                    'Metadata_' + args.model_name + '_split_'
                                                    + str(train_seed.item()) + '.txt'))
        with open(log_metadata, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('=== %s Metadata for Training/Testing (%s) ===\n ' % (args.model_name, now))
            for record in DataRegister.traintestmetadata:
                if 'IDN' in record:
                    message = f"\n\nTitle IDN: {record['IDN']}\n" \
                              f"\nTraining:\n" \
                              f"Month: {record['Train']['Month']};\nFrontPage: {record['Train']['FrontPages']};\nTotalPage: {record['Train']['TotalPages']};\nTrainRatio: {record['Train']['TrainRatio']}. \n" \
                              f"\nTesting:\n" \
                              f"Month: {record['Test']['Month']};\nFrontPage: {record['Test']['FrontPages']};\nTotalPage: {record['Test']['TotalPages']};\nTestRatio: {record['Test']['TestRatio']}. \n"
                    log_file.write(message)

        if args.test_ratio == 1:  # pure testing
            model.setup(niter=args.niter, niter_decay=args.niter_decay, epoch=args.test_epoch)
            model.eval()
            # create a folder to hold the evaluation results for all test images
            # in this folder create a text file to record test results
            logiter = 'epoch_%d' % args.test_epoch
            dir_independent_test = os.path.join(log_dir_testing, dir_model_seed, 'independent_test', logiter)
            if not os.path.exists(dir_independent_test):
                os.makedirs(dir_independent_test)
            result_independent_test = os.path.join(dir_independent_test, 'independent_test_' + logiter + '.txt')
            with open(result_independent_test, "a") as log_file:
                message = f"Result from Independet Testing \n\n" \
                          f"Test Model: {args.model_name} \n" \
                          f"Train Seed: {train_seed} \n" \
                          f"Train Epoch: {args.test_epoch}\n\n"
                log_file.write(message)
            print('Evaluating model on training seed ' + str(train_seed.item()) + ' epoch ' + str(args.test_epoch))

            testIDN = np.array([m['IDN'] for m in FPRTest.metadata])
            testYear = np.array([m['Year'] for m in FPRTest.metadata])
            testMonth = np.array([m['Month'] for m in FPRTest.metadata])
            testDate = np.array([m['Date'] for m in FPRTest.metadata])
            testPage = np.array([m['Page'] for m in FPRTest.metadata])

            testGT = np.array(FPRTest.label)
            testPred = -1 * np.ones(dataloaderTest.__len__())

            """
            in this for loop, all test data will be pushed through the trained network and the prediction
            as well as sample information will be collected
            """
            for i, data in enumerate(dataloaderTest):
                # take one image sample
                model.set_input(data)
                # run inference
                model.test()
                # retrive test batch output
                batchIDN = np.array(model.metadata['IDN'])
                batchYear = np.array(model.metadata['Year'])
                batchMonth = np.array(model.metadata['Month'])
                batchDate = np.array(model.metadata['Date'])
                batchPage = np.array(model.metadata['Page'])

                # retrieve prediction from the model
                batchPred = torch.nn.functional.softmax(model.pred, dim=1)
                batchPred = torch.topk(batchPred, k=1)
                predConfidence = batchPred[0].cpu().numpy().round(decimals=4)
                batchPred = batchPred[1].cpu().numpy()

                for j in torch.arange(start=0, step=1, end=len(batchPred)):
                    testPred[(testIDN == batchIDN[j]) &
                             (testYear == batchYear[j]) & (testMonth == batchMonth[j]) & (testDate == batchDate[j]) &
                             (testPage == batchPage[j])] = batchPred[j]

            # check that all samples are being processed
            if any(testPred == -1):
                raise Exception('Processing of test data unsuccessful')

            """
            Performance per title 
            """
            performanceMatrix = []
            errorLog = []
            with open(result_independent_test, "a") as log_file:
                log_file.write('\n%s \n\n' % "***Results per Title***")
            for title in DataRegister.titleIDN:
                [titlePrecision, titleRecall, titlebAccuracy, titleF1] = \
                    evaluation_metric(predictions=testPred[testIDN == title], labels=testGT[testIDN == title])

                message = f"\nTitleIDN: {title}\n" \
                          f"Precision: {titlePrecision}\n" \
                          f"Recall: {titleRecall}\n" \
                          f"bAccuracy: {titlebAccuracy}\n" \
                          f"titleF1: {titleF1}\n"

                with open(result_independent_test, "a") as log_file:
                    log_file.write(message)
            """
            Over all performance
            """
            [Precision, Recall, bAccuracy, F1] = evaluation_metric(predictions=testPred, labels=testGT)
            message = f"\nTitleIDN: All titles \n" \
                      f"Precision: {Precision}\n" \
                      f"Recall: {Recall}\n" \
                      f"bAccuracy: {bAccuracy}\n" \
                      f"titleF1: {F1}\n"
            with open(result_independent_test, "a") as log_file:
                log_file.write('\n%s \n\n' % "***Results All***")
                log_file.write(message)

            errorSamples = np.where(testPred != testGT)[0]
            if len(errorSamples) > 0:
                with open(result_independent_test, "a") as log_file:
                    log_file.write('\n%s \n\n' % "***Error Samples***\n")
                for errSample in errorSamples:
                    message = f"\nTitleIDN: {testIDN[errSample]} \n" \
                              f"Year: {testYear[errSample]}\n" \
                              f"Month: {testMonth[errSample]}\n" \
                              f"Date: {testDate[errSample]}\n" \
                              f"Page: {testPage[errSample]}\n" \
                              f"GT: {testGT[errSample]}\n" \
                              f"Prediction: {testPred[errSample]}\n" \
                              f"AttackLocation: {testAttackLocation[errSample]}"
                    with open(result_independent_test, "a") as log_file:
                        log_file.write(message)

        else:  # normal training
            model.setup(niter=args.niter, niter_decay=args.niter_decay, epoch=epoch_count - 1)
            # if the training is resumed from a previously trained model (indicated by epoch_count, the learning rate
            # will be updated accordingly
            if epoch_count > 1:
                model.update_learning_rate(epoch_lr=epoch_count)

            logger = Logger(checkpoints_dir=os.path.join(log_dir_training, dir_model_seed),
                            epoch_lr=epoch_count)

            for epoch in range(epoch_count, args.niter + args.niter_decay + 1):
                epoch_start_time = time.time()
                iter_data_time = time.time()
                epoch_iter = 0

                for i, data in enumerate(dataloaderTrain):
                    iter_start_time = time.time()

                    if total_iters % args.print_freq == 0:
                        t_data = iter_start_time - iter_data_time

                    logger.reset()
                    total_iters += args.batch_size
                    epoch_iter += args.batch_size

                    model.set_input(data)
                    model.optimize_parameters()

                    if total_iters % args.print_freq == 0:  # print training losses and save logging information to the disk
                        losses = model.get_current_losses()
                        t_comp = (time.time() - iter_start_time) / args.batch_size
                        logger.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

                print('End of epoch %d \t Time Taken: %s' % (
                    epoch, datetime.timedelta(seconds=(time.time() - epoch_start_time))))
                if epoch % args.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                    print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                    model.save_networks(epoch)

                model.update_learning_rate()  # update learning rates at the end of every epoch.

                """
                Test the model for every epoch
                """
                test_epoch = epoch
                model_test = FPRNet(checkpoints_dir=log_dir_training, model_name=dir_model_seed,
                                    num_class=args.num_class, gpu_ids=gpu_ids, isTrain=False)
                model_test.setup(niter=0, niter_decay=0, epoch=epoch, printnetwork=False)
                model_test.eval()
                # create a folder (for each epoch) to hold the evaluation results for all test images
                # in this folder create a text file to record test results
                logiter = 'epoch_%d' % test_epoch
                dir_intermediate_test = os.path.join(log_dir_testing, dir_model_seed, logiter)
                if not os.path.exists(dir_intermediate_test):
                    os.makedirs(dir_intermediate_test)
                log_result = os.path.join(dir_intermediate_test, str(train_seed.item()) + '_log_' + logiter + '.txt')
                with open(log_result, "a") as log_file:
                    now = time.strftime("%c")
                    log_file.write('==== %s Test Result (%s) ====\n '
                                   'Test Model: FPR epoch_%d\n' % (dir_model_seed, now, test_epoch))
                print('Evaluating model on train_seed ' + str(train_seed.item()) + ' epoch ' + str(test_epoch))

                testIDN = np.array([m['IDN'] for m in FPRTest.metadata])
                testYear = np.array([m['Year'] for m in FPRTest.metadata])
                testMonth = np.array([m['Month'] for m in FPRTest.metadata])
                testDate = np.array([m['Date'] for m in FPRTest.metadata])
                testPage = np.array([m['Page'] for m in FPRTest.metadata])
                testGT = np.array(FPRTest.label)
                testPred = -1 * np.ones(dataloaderTest.__len__())

                """
                in this for loop, all test data will be pushed through the trained network and the prediction
                as well as sample information will be collected
                """
                for i, data in enumerate(dataloaderTest):
                    # take one image sample
                    model_test.set_input(data)
                    # run inference
                    model_test.test()
                    # retrive test batch output
                    batchIDN = np.array(model_test.metadata['IDN'])
                    batchYear = np.array(model_test.metadata['Year'])
                    batchMonth = np.array(model_test.metadata['Month'])
                    batchDate = np.array(model_test.metadata['Date'])
                    batchPage = np.array(model_test.metadata['Page'])

                    # retrieve prediction from the model
                    batchPred = torch.nn.functional.softmax(model_test.pred, dim=1)
                    batchPred = torch.topk(batchPred, k=1)
                    # predConfidence = batchPred[0].cpu().numpy().round(decimals=4)
                    batchPred = batchPred[1].cpu().numpy()

                    for j in torch.arange(start=0, step=1, end=len(batchPred)):
                        testPred[(testIDN == batchIDN[j]) &
                                 (testYear == batchYear[j]) & (testMonth == batchMonth[j]) & (
                                         testDate == batchDate[j]) &
                                 (testPage == batchPage[j])] = batchPred[j]

                # check that all samples are being processed
                if any(testPred == -1):
                    raise Exception('Processing of test data unsuccessful')

                """
                Evaluation: per title and over all
                """
                performanceMatrix = []
                for title in DataRegister.titleIDN:
                    [titlePrecision, titleRecall, titlebAccuracy, titleF1] = \
                        evaluation_metric(predictions=testPred[testIDN == title], labels=testGT[testIDN == title])
                    performanceMatrix.append({'TitleIDN': title,
                                              'Precision': titlePrecision, 'Recall': titleRecall,
                                              'bAccuracy': titlebAccuracy, 'F1': titleF1})
                [Precision, Recall, bAccuracy, F1] = \
                    evaluation_metric(predictions=testPred, labels=testGT)
                performanceMatrix.append({'TitleIDN': 'All',
                                          'Precision': Precision, 'Recall': Recall,
                                          'bAccuracy': bAccuracy, 'F1': F1})

                message = '(Overall Precision: %.2f, Overall Recall: %.2f, Overall bAccuracy: %.2f, Overall F1: %.2f)' \
                          % (Precision, Recall, bAccuracy, F1)
                with open(log_result, "a") as log_file:
                    log_file.write('%s \n' % message)
