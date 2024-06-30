import numpy as np
import torch
import pdb

class Recorder:
    def __init__(self, verbose=False, delta=0, early_stop_time=7, warmup=0):
        self.verbose = verbose
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta
        self.decrease_time = 0
        self.early_stop_time = 3
        self.warmup = warmup

    def __call__(self, val_loss, model, path, epoch, early_stop=True):
        if epoch > self.warmup:
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path)
            elif score > self.best_score + self.delta:
                self.best_score = score
                self.save_checkpoint(val_loss, model, path)
                self.decrease_time = 0
            else:
                self.decrease_time += 1
        # print ("___________")
        # print (f'Epoch: {epoch}, val_loss: {val_loss}, score: {score}, best_score: {self.best_score}, decrease_time: {self.decrease_time}, early_stop_time: {self.early_stop_time}')
        # print (self.decrease_time >= self.early_stop_time if early_stop else 0)
        # print ("___________")
        return self.decrease_time >= self.early_stop_time if early_stop else 0
        #return True if early_stop else 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss
