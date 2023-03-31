import os
import time
import datetime
import argparse

"""
import for training/testing
"""
import torch
from data.base_dataset import OCTDataset, OCTDataLoader, data_query
from models.OCT_ResNeSt import OCTNet

"""
import for testing
"""
import numpy as np
from scipy.stats import spearmanr

from utils.utils import Visualizer
from utils.train_test import accuracy_calc

parser = argparse.ArgumentParser(description='Training on OCT images with featMap: underline model ResNeSt')

parser.add_argument('--model_name', default='OCT_ResNeSt', type=str, help='name of the model')
parser.add_argument('--data_dir', default=r'../data/TestBench2021/quality', type=str,
                    help='dir where to retrieve train and test data')
parser.add_argument('--save_dir', default=r'./checkpoints/training', type=str,
                    help='dir where to save models and images during training')
parser.add_argument('--result_dir', default=r'./checkpoints/testing', type=str,
                    help='dir where to save results during testing')

parser.add_argument('--data_mode', default='s', type=str,
                    help='how to read in the OCT images, usage: a (all), s (stacked), f (fused)')
parser.add_argument('--num_class', default=4, type=int, help='number of classes')
parser.add_argument('--batch_size', default=3, type=int, help='batch size for training')
parser.add_argument('--label_mode', default=0, type=int, help='which label to use, usage: 0 (quality), 1 (pregnancy)')

parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--train_device', default='cuda', type=str, help='cpu or cuda')

args = parser.parse_args()

if args.train_device == 'cpu':
    gpu_ids = []
else:
    gpu_ids = [0]

imgH = 800
imgW = 800

"""
Training scheme configuration
"""
niter = 50
niter_decay = 50

print_freq = 1 * args.batch_size
save_latest_freq = 100 * args.batch_size
save_epoch_freq = 1

"""
Testing scheme configuration
"""

if args.label_mode == 0:
    labeltype = 'quality'
elif args.label_mode == 1:
    labeltype = 'pregnancy'

repstrt = 1
if __name__ == '__main__':
    if args.train_device is 'cuda':
        torch.cuda.empty_cache()

    for datasetrep in torch.arange(start=repstrt, step=1, end=4):

        total_iters = 0
        epoch_count = 1
        if datasetrep > repstrt:
            epoch_count = 1

        # create train dataloader
        datasetname = os.path.join(args.data_dir, str(datasetrep.item()))
        datasetTrain = OCTDataset(datasetname=datasetname, mode='train', datamode=args.data_mode,
                                  labelmode=args.label_mode)
        dataloaderTrain = OCTDataLoader(dataset=datasetTrain, datamode=args.data_mode, batchsize=args.batch_size,
                                        num_threads=args.num_workers)
        print('Training', ': the number of training images = %d' % dataloaderTrain.__len__())

        # create test dataloader
        datasetTest = OCTDataset(datasetname=datasetname, mode='test', datamode=args.data_mode,
                                 labelmode=args.label_mode)
        dataset_size = len(datasetTest)
        dataloaderTest = OCTDataLoader(dataset=datasetTest, num_threads=args.num_workers)
        print('test', ': the number of testing images = %d' % (dataloaderTest.__len__()))
        results_dir_rep = os.path.join(args.result_dir, args.model_name, str(datasetrep.item()))

        # create model
        model_name_rep = os.path.join(args.model_name, str(datasetrep.item()))
        model = OCTNet(checkpoints_dir=args.save_dir, model_name=model_name_rep, num_class=args.num_class,
                       gpu_ids=gpu_ids)
        model.setup(niter=niter, niter_decay=niter_decay, epoch=epoch_count - 1)

        if epoch_count > 1:
            model.update_learning_rate(epoch_lr=epoch_count)

        visualizer = Visualizer(checkpoints_dir=os.path.join(args.save_dir, model_name_rep), epoch_lr=epoch_count)

        for epoch in range(epoch_count, niter + niter_decay + 1):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            for i, data in enumerate(dataloaderTrain):
                iter_start_time = time.time()

                if total_iters % print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                visualizer.reset()
                total_iters += args.batch_size
                epoch_iter += args.batch_size

                model.set_input(data)
                model.optimize_parameters()

                if total_iters % print_freq == 0:  # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / args.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

                if total_iters % save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters
                    model.save_networks(save_suffix)

                iter_data_time = time.time()

            if epoch % save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %s' % (epoch, niter + niter_decay,
                                                              datetime.timedelta(
                                                                  seconds=(time.time() - epoch_start_time))))
            model.update_learning_rate()  # update learning rates at the end of every epoch.

            """
            Test the model for every epoch
            """
            test_epoch = epoch
            model_test = OCTNet(checkpoints_dir=args.save_dir, model_name=model_name_rep,
                                num_class=args.num_class, gpu_ids=gpu_ids, isTrain=False)
            model_test.setup(niter=0, niter_decay=0, epoch=epoch, printnetwork=False)
            model_test.eval()
            # create a folder (for each epoch) to hold the evaluation results for all test images
            # in this folder create a text file to record test results
            logiter = 'epoch_%d' % test_epoch
            results_dir_iter = os.path.join(results_dir_rep, logiter)
            if not os.path.exists(results_dir_iter):
                os.makedirs(results_dir_iter)
            log_result = os.path.join(results_dir_iter, str(datasetrep.item()) + '_log_' + logiter + '.txt')
            with open(log_result, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('==== %s Test Result (%s) ====\n '
                               'Test Model: OCT epoch_%d\n' % (model_name_rep, now, test_epoch))
            print('Evaluating model on split ' + str(datasetrep.item()) + ' epoch ' + str(test_epoch))

            patientID = -1 * np.ones(dataloaderTest.__len__()).astype(int)
            wellNb = -1 * np.ones(dataloaderTest.__len__()).astype(int)
            focalLen = -1 * np.ones(dataloaderTest.__len__()).astype(int)

            # create a single ID for each different oocyte
            sampleID = -1 * np.ones(dataloaderTest.__len__()).astype(int)

            qualityGT = -1 * np.ones(dataloaderTest.__len__())
            qualityPred = -1 * np.ones(dataloaderTest.__len__())
            pregnancyGT = -1 * np.ones(dataloaderTest.__len__())
            pregnancyPred = -1 * np.ones(dataloaderTest.__len__())
            predConfidence = -1 * np.ones(dataloaderTest.__len__())

            """
            in this for loop, all test data will be pushed through the trained network and the prediction
            as well as sample information will be collected
            """
            for i, data in enumerate(dataloaderTest):
                # take one image sample
                model.set_input(data)
                # run inference
                model.test()
                # retrive sample information
                sampleinfo = data_query(data, args.data_mode)

                if args.label_mode == 0:
                    model_label = model.label + 1
                elif args.label_mode == 1:
                    model_label = model.label
                if sampleinfo[labeltype] != model_label:
                    raise Exception('Inconsistent reading of label of data')
                else:
                    patientID[i] = sampleinfo['patientID']
                    wellNb[i] = sampleinfo['well']
                    sampleID[i] = int(str(patientID[i]) + str(wellNb[i]))
                    focalLen[i] = sampleinfo['FL']
                    qualityGT[i] = sampleinfo['quality']
                    pregnancyGT[i] = sampleinfo['pregnancy']

                    # retrieve prediction from the model
                    qualityPredtemp = torch.nn.functional.softmax(model.pred, dim=1)
                    qualityPredtemp = torch.topk(qualityPredtemp, k=1)
                    predConfidence[i] = qualityPredtemp[0].cpu().numpy().round(decimals=4)
                    if args.label_mode == 0:
                        qualityPred[i] = qualityPredtemp[1].cpu().numpy() + 1
                    elif args.label_mode == 1:
                        pregnancyPred[i] = qualityPredtemp[1].cpu().numpy()

            # check that all samples are being processed
            if any(sampleID == -1) or \
                    any((qualityPred == -1) & (pregnancyPred == -1)):
                raise Exception('Load of test data unsuccessful')

            """
            Prepare the ground-truth for evaluation on the prediction accuracy, lcc and srocc
            one oocyte sample -> one well number -> 11 predictions (if args.data_mode is 'a') or
            1 prediction (if args.data_mode is 's') -> one patient ID
            """
            # get sample IDs, prediction will be averaged if more than one prediction
            # is performed on one oocyte (for example when args.data_mode is 'a')
            imglist = np.unique(sampleID)
            img_ID = -1 * np.ones(imglist.size)
            img_well = -1 * np.ones(imglist.size)
            img_FL = -1 * np.ones(imglist.size)
            img_tar = -1 * np.ones(imglist.size)
            img_pred = -1 * np.ones(imglist.size)
            img_predConf = -1 * np.ones(imglist.size)
            img_predFL = -1 * np.ones(imglist.size)

            # if args.data_mode is 'a', a full copy of the prediction
            # on all 11 FLs will be saved
            if args.data_mode is 'a':
                img_predRecord = -1 * np.ones((imglist.size, 3, 11), dtype=np.int64)

            # the number of checksum should equal the number of total predictions
            # in this for loop, predictions will be prepared. When args.data_mode is 'a'
            # the predictions on 11 FLs of a same oocyte are averaged to produce the final
            # prediction
            checksum = 0
            for i in range(len(imglist)):
                imgIDtemp = imglist[i]
                # get all predictions correspond to this sample
                retriveMarker = (sampleID == imgIDtemp)
                checksum += sum(retriveMarker)

                # 11 different FLs of a sample should correspond to a same patient ID
                patientcheck = np.unique(patientID[retriveMarker])
                if patientcheck.size != 1:
                    raise Exception('Inconsistent reading of patientID of data')
                else:
                    img_ID[i] = patientcheck

                # 11 different FLs of a sample should correspond to a same well number
                wellcheck = np.unique(wellNb[retriveMarker])
                if wellcheck.size != 1:
                    raise Exception('Inconsistent reading of patientID of data')
                else:
                    img_well[i] = wellcheck

                # one sample should correspond 11 or 1 image
                FLcheck = np.unique(focalLen[retriveMarker])
                if not (FLcheck.size == 1 or FLcheck.size == 11):
                    raise Exception('Inconsistent reading of focal length of data')
                else:
                    img_FL[i] = FLcheck

                if args.label_mode == 0:
                    img_tar_temp = np.unique(qualityGT[retriveMarker])
                elif args.label_mode == 1:
                    img_tar_temp = np.unique(pregnancyGT[retriveMarker])

                # the ground truth of different images correspond to a same oocyte
                # must be the same
                if img_tar_temp.size != 1:
                    raise Exception('Inconsistent reading of quality of data')
                else:
                    img_tar[i] = img_tar_temp

                # when args.data_mode is 'a', one oocyte has 11 different predictions, where
                # the final prediction is the average of all predictions
                # when args.data_mode is 's', one oocyte has 1 prediction
                if args.label_mode == 0:
                    tarPredtemp = qualityPred[retriveMarker]
                elif args.label_mode == 1:
                    tarPredtemp = pregnancyPred[retriveMarker]
                # img_pred[i] = np.mean(tarPredtemp).round(0)
                # aternatively, we can log the prediction with the highest confidence and the FL
                # on which the prediction is being made
                tarConftemp = predConfidence[retriveMarker]
                predPosition = np.argmax(tarConftemp)
                img_pred[i] = tarPredtemp[predPosition]
                img_predConf[i] = tarConftemp[predPosition]
                img_predFL[i] = focalLen[retriveMarker][predPosition]

                # log the ground-truth and prediction infomration
                if args.label_mode == 0:
                    message = 'patientID: %d, wellNbr: %d,  quality: %d, Prediction: %d ' \
                              % (patientcheck, wellcheck, img_tar[i], img_pred[i])
                elif args.label_mode == 1:
                    message = 'patientID: %d, wellNbr: %d,  pregnancy: %d, Prediction: %d ' \
                              % (patientcheck, wellcheck, img_tar[i], img_pred[i])

                with open(log_result, "a") as log_file:
                    log_file.write('%s \n' % message)

            """
             after the for loop: check that all samples are processed, log testing information
            """
            if (any((img_tar == -1) | (img_pred == -1))
                    or any((img_ID == -1) | (img_well == -1) | (img_FL == -1))
                    or checksum != dataloaderTest.__len__()):
                raise Exception('Parse of test data unsuccessful')
            else:
                accuracy = accuracy_calc(img_pred, img_tar)
                lcc = np.corrcoef(img_pred, img_tar)[1, 0]
                srocc, _ = spearmanr(img_pred, img_tar)

                message = '(accuracy: %.2f, lcc: %.4f, srocc: %.4f)' % (accuracy, lcc, srocc)
                with open(log_result, "a") as log_file:
                    log_file.write('%s \n' % message)