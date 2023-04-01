import os
import time


# A simple loss logger
class Logger:

    def __init__(self, checkpoints_dir, name='Experiment_Log', epoch_lr=1):

        # create a logging file to store training losses
        self.log_name = os.path.join(checkpoints_dir, name, 'loss_log.txt')
        if not os.path.exists(os.path.join(checkpoints_dir, name)):
            os.makedirs(os.path.join(checkpoints_dir, name))
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            if epoch_lr == 1:
                log_file.write('================ Training Loss (%s) ================\n' % now)
            else:
                log_file.write('================ Training Loss Cont (%d) ================\n' % epoch_lr)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
