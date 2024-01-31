from constants import IMAGE_KEY, LABEL_KEY
import torch


class Trainer:
    def __init__(self, network, optimizer, loss_function, train_loader, validation_loader):
        self.network = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.validation_loader = validation_loader

    def train_step(self):
        self.network.train()
        total_train_loss = 0
        for batch_data in self.train_loader:
            x, y = batch_data[IMAGE_KEY], batch_data[LABEL_KEY]
            prediction = self.network(x)
            loss = self.loss_function(prediction.to(torch.float32), y.to(torch.float32))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_train_loss += loss
        return total_train_loss

    def evaluate_step(self):
        total_validation_loss = 0
        with torch.no_grad():
            self.network.eval()
            for batch_data in self.validation_loader:
                x, y = batch_data[IMAGE_KEY], batch_data[LABEL_KEY]
                prediction = self.network(x)
                total_validation_loss += self.loss_function(prediction.to(torch.float32), y.to(torch.float32))
        return total_validation_loss
