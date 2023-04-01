import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import os.path
from collections import OrderedDict


class BaseModel(nn.Module):

    def __init__(self, checkpoints_dir, model_name, isTrain=True, gpu_ids=[]):

        super().__init__()
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        self.gpu_ids = gpu_ids
        self.save_dir = os.path.join(os.path.abspath(checkpoints_dir), model_name)

        self.isTrain = isTrain

        self.loss_names = []
        self.model_names = []
        self.optimizers = []
        self.image_paths = []

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def setup(self, niter=0, niter_decay=0, load_iter=0, epoch=0, printnetwork=True):
        """Load and print networks; create schedulers
        """
        if self.isTrain:
            self.schedulers = [self.get_scheduler(optimizer, 'linear', niter=niter, niter_decay=niter_decay)
                               for optimizer in self.optimizers]
        if not self.isTrain or load_iter > 0 or epoch > 0:
            load_suffix = 'iter_%d' % load_iter if load_iter > 0 else epoch
            self.load_networks(load_suffix)

        if printnetwork:
            self.print_networks()

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()

    def update_learning_rate(self, epoch_lr=None):
        """Update learning rates for all the networks; called at the end of every epoch"""
        if epoch_lr is not None:
            print('Mannually update _step_count in scheduler(s)')
            for scheduler in self.schedulers:
                scheduler._step_count = epoch_lr

        for scheduler in self.schedulers:
            scheduler.step(epoch=epoch_lr)

        for i in torch.arange(0, len(self.optimizers)):
            lr = self.optimizers[i].param_groups[0]['lr']
            print(self.model_names[i], ': learning rate = %.7f' % lr)

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    try:
                        torch.save(net.module.cpu().state_dict(), save_path)
                        net.cuda(self.gpu_ids[0])
                    except:
                        torch.save(net.cpu().state_dict(), save_path)
                        net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                try:
                    state_dict = torch.load(load_path, map_location=self.device)
                except:
                    state_dict = torch.jit.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)

    def print_networks(self):
        """Print the total number of parameters in the network and network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    @staticmethod
    def get_scheduler(optimizer, lr_policy, niter, niter_decay):
        """Return a linear learning rate scheduler

        Parameters:
            optimizer          -- the optimizer of the network

        For 'linear', we keep the same learning rate for the first <niter> epochs
        and linearly decay the rate to zero over the next <niter_decay> epochs.
        For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
        See https://pytorch.org/docs/stable/optim.html for more details.
        """
        if lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 - niter) / float(niter_decay + 1)
                return lr_l

            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
        return scheduler

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
