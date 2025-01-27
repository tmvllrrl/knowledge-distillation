import os
import json
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torchvision.models import resnet18, resnet34, ResNet34_Weights

class BaselineConfig():
    def __init__(self, save_dir: str) -> None:
        # Used to auto save config
        self.save_dir = save_dir

        # Hyperparameters
        self.lr: float = 1e-4
        self.epochs: int = 10
        self.batch_size: int = 128
        self.num_workers: int = 2

        self.classifier_name: str = "ResNet18"
        self.criterion_name: str = "CrossEntropyLoss"
        self.optimizer_name: str = "Adam"

        self.dataset_name: str = "CIFAR10" 

        self.init_objects()
        self.save_config()

    def init_objects(self) -> None:
        # Initializing classifier and changing FC for 10 classes
        self.classifier = resnet18(weights=None)
        self.classifier.fc = nn.Linear(512, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.params = [
            {"params": self.classifier.parameters(), "lr": self.lr}
        ]
        self.optimizer = Adam(self.params)

        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

        self.train_dataset = CIFAR10(root='./data', train=True, transform=train_transforms)

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=self.num_workers
        )    

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])

        # CIFAR10 does not have a validation split
        self.test_dataset = CIFAR10(root='./data', train=False, transform=test_transforms)
        
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers
        )    

    def save_config(self) -> None:

        config = {
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "classifier": self.classifier_name,
            "criterion": self.criterion_name,
            "optimizer": self.optimizer_name,
            "dataset": self.dataset_name
        }

        config_save_path = os.path.join(self.save_dir, 'config.json')

        with open(config_save_path, "w") as file:
            json.dump(config, file, indent=4)


class KDConfig():
    def __init__(self, save_dir: str) -> None:
        # Used to auto save config
        self.save_dir = save_dir

        # Hyperparameters
        self.lr: float = 1e-4
        self.epochs: int = 10
        self.batch_size: int = 128
        self.num_workers: int = 2

        self.teacher_name: str = "ResNet34"
        self.student_name: str = "ResNet18"
        self.criterion_name: str = "CrossEntropyLoss"
        self.optimizer_name: str = "Adam"

        self.dataset_name: str = "CIFAR10" 

        self.init_objects()
        self.save_config()

    def init_objects(self) -> None:
        # Initializing teacher and student and changing FC for 10 classes
        self.teacher = resnet34(weights=None)
        self.teacher.fc = nn.Linear(512, 10)
        # Assuming we have a pre-trained teacher model
        # self.teacher.load_state_dict(torch.load('blah', weights_only=True))

        self.student = resnet18(weights=None)
        self.student.fc = nn.Linear(512, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.params = [
            {"params": self.student.parameters(), "lr": self.lr}
        ]
        self.optimizer = Adam(self.params)

        base_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.train_dataset = CIFAR10(root='./data', train=True, transform=base_transforms)

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=self.num_workers
        )    

        # CIFAR10 does not have a validation split
        self.test_dataset = CIFAR10(root='./data', train=False, transform=base_transforms)
        
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers
        )    

    def save_config(self) -> None:

        config = {
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "teacher": self.teacher_name,
            "student": self.student_name,
            "criterion": self.criterion_name,
            "optimizer": self.optimizer_name,
            "dataset": self.dataset_name
        }

        config_save_path = os.path.join(self.save_dir, 'config.json')

        with open(config_save_path, "w") as file:
            json.dump(config, file, indent=4)
