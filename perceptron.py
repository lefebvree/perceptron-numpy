
from network import Network
import numpy as np


class Perceptron(Network):

    def __init__(self):
        super().__init__()

        # Taux d'apprentissage
        self.TRAIN_STEP = .00005

    def initialize_network(self):
        # Initialisation des poids aléatoires, on divise par SIZE_INPUT pour ne pas avoir des valeurs trop importantes
        self.weights = np.random.rand(self.SIZE_OUTPUT, self.SIZE_INPUT + 1) / self.SIZE_INPUT

    def train(self, image, label):
        input_data = np.append(image, 1)
        outputs = self.get_outputs(image)

        # Modification des poids pour chaque neuronne
        for i in range(self.SIZE_OUTPUT):
            self.weights[i] += self.TRAIN_STEP * input_data * (label[i] - outputs[i])

    def get_outputs(self, inputs):
        # Ajout de l'entrée virtuelle à 1.0
        input_data = np.append(inputs, 1)
        return np.dot(self.weights, input_data)

    def print_params(self):
        print(" • η =", self.TRAIN_STEP)


if __name__ == '__main__':
    perceptron = Perceptron()
    perceptron.dataset_train()
    perceptron.test()
