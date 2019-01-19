
from abc import ABC, abstractmethod
import pickle
import gzip

import torch
import torch.utils.data
import numpy as np


class Network(ABC):

    def __init__(self):
        self.SIZE_INPUT = 28 * 28
        self.SIZE_OUTPUT = 10

        self.TRAIN_BATCH_SIZE = 1
        self.TEST_BATCH_SIZE = 1

        self.train_loader = None
        self.test_loader = None

        self.weights = None

    def initialize_train_data(self, path='mnist_light_CN.pkl.gz'):
        # Données
        data = pickle.load(gzip.open(path), encoding='latin1')
        # Images de la base d'apprentissage
        train_data = torch.Tensor(data[0][0])
        # Labels de la base d'afor image,label in train_loader:
        train_data_label = torch.Tensor(data[0][1])
        # Images de la base de test
        test_data = torch.Tensor(data[1][0])
        # Labels de la base de test
        test_data_label = torch.Tensor(data[1][1])
        # Base de données d'apprentissage (pour torch)
        train_dataset = torch.utils.data.TensorDataset(train_data, train_data_label)
        # Base de données de test (pour torch)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_data_label)
        # Lecteur de la base de données d'apprentissage (pour torch)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.TRAIN_BATCH_SIZE, shuffle=True)
        # Lecteur de la base de données de test (pour torch)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.TEST_BATCH_SIZE , shuffle=False)

    @abstractmethod
    def initialize_network(self):
        ...

    def dataset_train(self, steps=25000):
        # Initialise les poids et les données d'apprentissage si ce n'est deja fait
        if self.weights is None:
            self.initialize_network()

        if self.train_loader is None:
            self.initialize_train_data()

        print("Parameters :")
        print(" • steps =", steps)
        self.print_params()
        print()

        # steps représente le nombre de données d'apprentissage à parcourir
        progress = 0

        # Pour chaque image dans le jeu d'entrainement
        while progress < steps:
            for image, label in self.train_loader:
                progress += 1

                print("[    ] Training [", progress, "/", steps, "]...", end="\r")
                self.train(image.flatten().numpy(), label[0].numpy())

                # On arrete au bout de steps étapes
                if progress >= steps:
                    break
        print("[DONE] Training [", progress, "/", steps, "]   ")

    @abstractmethod
    def train(self, image, label):
        ...

    @abstractmethod
    def get_outputs(self, inputs):
        ...

    def test(self):
        # Parcours le jeu de test et affiche la précision

        if self.test_loader is None:
            self.initialize_train_data()

        answers_total = 0
        answers_correct = 0

        for image, label in self.test_loader:
            answers_total += 1
            print("[    ] Testing  [", answers_total, "]...", end="\r")

            input_data = image.flatten().numpy()

            expected = np.argmax(label).item()
            outputs = self.get_outputs(input_data)

            if type(outputs) is np.ndarray:
                answer = np.argmax(outputs)
            else:
                answer = np.argmax(outputs.detach().numpy())

            if expected == answer:
                answers_correct += 1

        precision = answers_correct / answers_total * 100
        print("[DONE] Testing  [", answers_correct, "correct /", answers_total, "]")
        print()
        print("Precision : {0:0.2f} %".format(precision))

    @abstractmethod
    def print_params(self):
        ...
