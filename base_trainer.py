import os
import torch
import torchvision
import torch.nn as nn

from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights

from configs import *
from util import KnowledgeDistilSoftmax, SoftCrossEntropyLoss

class Trainer():
    def __init__(
            self,
            classifier: torch.nn.Module,
            epochs: int,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            train_dataloader: torch.utils.data.DataLoader,
            val_dataloader: torch.utils.data.DataLoader,
            save_dir: str,
        ) -> None:  

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = save_dir
        
        self.classifier = classifier
        self.classifier = self.classifier.to(self.device)

        # Set up necessary items for training
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer

        # Set up train and validation dataloaders
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.writer = SummaryWriter(self.save_dir)

    def train(self) -> None:

        for ep in (pbar := tqdm(range(self.epochs))):
            self.classifier.train()

            train_running_loss = 0
            train_correct = 0
            train_total = 0

            for bi, (image, label) in enumerate(self.train_dataloader):
                image = image.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()

                logits = self.classifier(image)

                loss = self.criterion(logits, label)
                loss.backward()

                self.optimizer.step()

                train_running_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == label).sum().item()
                train_total += len(label)

            train_loss = train_running_loss / len(self.train_dataloader)
            train_accuracy = train_correct / train_total

            valid_loss, valid_accuracy = self.validate()

            self.writer.add_scalar('Loss/train', train_loss, ep+1)
            self.writer.add_scalar('Accuracy/train', train_accuracy, ep+1)
            self.writer.add_scalar('Loss/valid', valid_loss, ep+1)
            self.writer.add_scalar('Accuracy/valid', valid_accuracy, ep+1)

            pbar.set_description(f"EPOCH {ep+1} TRAIN LOSS: {train_loss:.4f}")

            torch.save(self.classifier.state_dict(), f'{self.save_dir}/classifier_{ep+1}.pt')

    def validate(self) -> tuple[float, float]:
        self.classifier.eval()

        valid_running_loss = 0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for bi, (image, label) in enumerate(self.val_dataloader):
                image = image.to(self.device)
                label = label.to(self.device)

                logits = self.classifier(image)

                loss = self.criterion(logits, label)
                valid_running_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                valid_correct += (predicted == label).sum().item()
                valid_total += len(label)

            valid_loss = valid_running_loss / len(self.val_dataloader)
            valid_accuracy = valid_correct / valid_total

        return valid_loss, valid_accuracy
    

def main() -> None:
    # Using time for unique save directory name
    save_dir = os.path.join("./runs/", datetime.now().strftime("%Y%b%d_%H:%M:%S"))
    os.makedirs(save_dir, exist_ok=True)

    config = BaselineConfig(save_dir=save_dir)

    trainer = Trainer(
        classifier=config.classifier,
        epochs=config.epochs,
        criterion=config.criterion,
        optimizer=config.optimizer,
        train_dataloader=config.train_dataloader,
        val_dataloader=config.test_dataloader,
        save_dir=config.save_dir
    )
    trainer.train()


if __name__ == "__main__":
    main()