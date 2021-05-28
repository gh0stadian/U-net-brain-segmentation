import csv
import torch
from torch import optim
from monai.losses.dice import DiceLoss
import datetime
from reload_staged_model import transfer_loaded_checkpoint_to_model


class Train:

    def __init__(self, n_of_epoch, train_loader, valid_loader, model, config):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.criterion = DiceLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.n_of_epoch = n_of_epoch
        self.log = []

        # LR TUNING
        self.n_epochs_decrease_lr = config["n_epoch_to_decrease_lr"]
        self.epochs_no_improve_lr = 0

        # EARLY STOPPING
        self.n_epochs_stop = config["n_epoch_to_stop"]
        self.epochs_no_improve_es = 0
        self.min_eval_loss = 10000.0
        self.starting_epoch = 0

        self.time_stamp = str(datetime.datetime.now().strftime("%d_%m_%Y(%H_%M_%S)"))

        if config.get("checkpoint_patch") is not None:
            print("RELOADING MODEL")
            self.model, self.optimizer, self.starting_epoch = \
                transfer_loaded_checkpoint_to_model(self.model,
                                                    self.optimizer,
                                                    str(config["checkpoint_patch"]))
            print("MODEL RELOADED")

        self.loop()

    def loop(self):
        print("TRAIN LOOP STARTED")
        for epoch in range(self.starting_epoch, self.n_of_epoch):
            # INIT DATASET ITERATORS
            train_iterator = iter(self.train_loader)
            valid_iterator = iter(self.valid_loader)

            # TRAIN
            train_mean_loss = self.train(train_iterator) / len(self.train_loader)

            # VALID
            eval_mean_loss = self.valid(epoch, valid_iterator) / len(self.valid_loader)

            # CHECK PROGRESS
            self.print_n_save_progress(epoch, train_mean_loss, eval_mean_loss)
            self.check_if_eval_improve(eval_mean_loss)

            # STAGE MODEL
            self.save_model_stage(epoch)

            # LR TUNING
            if self.n_epochs_decrease_lr == self.epochs_no_improve_lr:
                self.epochs_no_improve_lr = 0
                for param in self.optimizer.param_groups:
                    param['lr'] *= 0.5

            # EARLY STOPPING
            if self.n_epochs_stop == self.epochs_no_improve_es:
                print("EARLY STOPPING AT EPOCH: " + str(epoch))
                break

        # self.save_model()
        save_log_to_csv(self.log)
        print("TRAINING SUCCESFULLY ENDED")

    def train(self, train_iterator):
        self.model.train()
        train_loss_sum = 0

        for batch in train_iterator:
            self.optimizer.zero_grad()
            model_output = self.model.forward(batch['input'])
            loss = self.criterion(model_output, batch['output'])
            train_loss_sum += loss.item()
            loss.backward()
            self.optimizer.step()

        return train_loss_sum

    def valid(self, epoch, valid_iterator):
        self.model.eval()
        eval_loss_sum = 0

        for batch in valid_iterator:
            model_output = self.model.forward(batch['input'])
            loss = self.criterion(model_output, batch['output'])
            eval_loss_sum += loss.item()

        return eval_loss_sum

    def save_model_stage(self, epoch):
        PATH = "outputs/" + str(self.time_stamp) + str(epoch) + ".pth"
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion,
        }, PATH)

    def print_n_save_progress(self, epoch, train_mean_loss, eval_mean_loss):
        print("EPOCH " + str(epoch) + ">\t" +
              "TRAIN>" + "{:.8f}".format(train_mean_loss) + ", " +
              "EVAL>" + "{:.8f}".format(eval_mean_loss) + ", " +
              "LR>" + "{:.6f}".format(self.optimizer.param_groups[0]['lr']))
        self.log.append([epoch, train_mean_loss, eval_mean_loss])
        save_log_to_csv(self.log, self.time_stamp)

    def check_if_eval_improve(self, eval_mean_loss):
        if eval_mean_loss < self.min_eval_loss:
            self.epochs_no_improve_es = 0
            self.epochs_no_improve_lr = 0
            self.min_eval_loss = eval_mean_loss
        else:
            self.epochs_no_improve_es += 1
            self.epochs_no_improve_lr += 1


def save_log_to_csv(log, name):
    path = "progress_log/" + name + ".csv"
    head = ["epoch", "train_loss", "valid_loss"]
    with open(path, 'w') as f:
        write = csv.writer(f)
        write.writerow(head)
        write.writerows(log)
