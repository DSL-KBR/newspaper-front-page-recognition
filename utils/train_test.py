import os
from matplotlib import pyplot as plt

import numpy as np
from models.FPR_ResNeSt import FPRNet
from matplotlib import pyplot as plt
import torch


def evaluation_metric(predictions, labels):
    """
    Performance evaluation based on F-measures
    """
    TP = np.sum((labels == 1) & (predictions == 1))
    TN = np.sum((labels == 0) & (predictions == 0))
    FP = np.sum((labels == 0) & (predictions == 1))
    FN = np.sum((labels == 1) & (predictions == 0))

    precision = (TP / (TP + FP)).round(decimals=3)
    recall = (TP / (TP + FN)).round(decimals=3)  # TPR
    specificity = (TN / (TN + FP)).round(decimals=3)  # TNR

    bAccuracy = ((recall + specificity) / 2).round(decimals=3)
    f1 = (2 * TP / (2 * TP + FP + FN)).round(decimals=3)

    return precision, recall, bAccuracy, f1


def evaluation_visual(model, save_dir):
    assert isinstance(model, FPRNet), 'evaluation_visual is only developed for FPRnet'
    save_dir_img = os.path.join(save_dir, 'img')
    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)

    samples = model.sample.cpu()
    for i in torch.arange(start=0, step=1, end=samples.shape[0]):
        sample_img = model.sample[i, :, :, :].permute([1, 2, 0]).cpu()
        plt.rcParams["figure.figsize"] = [10.24, 10.24]
        plt.imshow(sample_img)
        plt.savefig(os.path.join(save_dir_img, f"{model.metadata['IDN'][i]}_"
                                                        f"{model.metadata['Year'][i]}_"
                                                        f"{model.metadata['Month'][i]}_"
                                                        f"{model.metadata['Date'][i]}_"
                                                        f"{model.metadata['Page'][i]}.jpg"))

    # # interpret ResNeSt outputs
    # decisions_imgnet = torch.topk(torch.nn.functional.softmax(model.featResNeSt, dim=1), dim=1, k=1)
    # decisions_imgnet_conf = decisions_imgnet[0].squeeze().cpu().numpy().round(decimals=2)
    # decisions_imgnet = decisions_imgnet[1].cpu().numpy()
    # # interpret Classifier outputs
    # decisions_Classifier = torch.topk(torch.nn.functional.softmax(model.netClassifier.featLinear, dim=1), dim=1, k=1)
    # decisions_Classifier_conf = decisions_Classifier[0].squeeze().cpu().numpy().round(decimals=2)
    # decisions_Classifier = decisions_Classifier[1].cpu().numpy()
    #
    # x_step = np.arange(0, samples.shape[2], samples.shape[2] / model.featResNeSt.shape[2]) + \
    #          samples.shape[2] / (2 * model.featResNeSt.shape[2])
    # y_step = np.arange(0, samples.shape[3], samples.shape[3] / model.featResNeSt.shape[3]) + \
    #          samples.shape[3] / (2 * model.featResNeSt.shape[3])
    #
    # for i in torch.arange(start=0, step=1, end=samples.shape[0]):
    #     sample_img = model.sample[i, :, :, :].permute([1, 2, 0]).cpu()
    #     plt.rcParams["figure.figsize"] = [10.24, 10.24]
    #     plt.imshow(sample_img)
    #     for x_loc_decision, x_loc_plot in enumerate(x_step):
    #         for y_loc_decision, y_loc_plot in enumerate(y_step):
    #             plt.text(x_loc_plot, y_loc_plot, f"class: {decisions_imgnet[i, 0, x_loc_decision, y_loc_decision]} \n" +
    #                      r'conf: %.2f' % decisions_imgnet_conf[i, x_loc_decision, y_loc_decision],
    #                      bbox=dict(facecolor='red', alpha=0.8), fontsize=20, fontweight="bold",
    #                      horizontalalignment='center', verticalalignment='center')
    #     plt.savefig(os.path.join(model.save_dir, 'img', f"{model.metadata['IDN'][i]}_"
    #                                                     f"{model.metadata['Year'][i]}_"
    #                                                     f"{model.metadata['Month'][i]}_"
    #                                                     f"{model.metadata['Date'][i]}_"
    #                                                     f"{model.metadata['Page'][i]}.jpg"))
    #     plt.show()
    #
    # for i in torch.arange(start=0, step=1, end=samples.shape[0]):
    #     sample_img = model.sample[i, :, :, :].permute([1, 2, 0]).cpu()
    #     plt.rcParams["figure.figsize"] = [10.24, 10.24]
    #     plt.imshow(sample_img)
    #     # ax = plt.axes()
    #     # ax.axis('off')
    #     for x_loc_decision, x_loc_plot in enumerate(x_step):
    #         for y_loc_decision, y_loc_plot in enumerate(y_step):
    #             plt.text(x_loc_plot, y_loc_plot,
    #                      f"class: {decisions_Classifier[i, 0, x_loc_decision, y_loc_decision]} \n" +
    #                      r'conf: %.2f' % decisions_Classifier_conf[i, x_loc_decision, y_loc_decision],
    #                      bbox=dict(facecolor='red', alpha=0.8), fontsize=20, fontweight="bold",
    #                      horizontalalignment='center', verticalalignment='center')
    #     plt.savefig(os.path.join(model.save_dir, 'img', f"{model.metadata['IDN'][i]}_"
    #                                                     f"{model.metadata['Year'][i]}_"
    #                                                     f"{model.metadata['Month'][i]}_"
    #                                                     f"{model.metadata['Date'][i]}_"
    #                                                     f"{model.metadata['Page'][i]}.jpg"))
    #     plt.show()
