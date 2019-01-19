
from network import Network
import numpy as np
import math


class Shallow(Network):

    def __init__(self):
        super().__init__()

        # Taux d'apprentissage couche cachée
        self.TRAIN_STEP_HIDDEN = 0.15

        # Taux d'apprentissage couche de sortie
        self.TRAIN_STEP_OUTPUT = 0.005

        # Taille de la couche cachée
        self.SIZE_HIDDEN = 150

        # Fonction sigmoide appliquée à la sortie de la couche cachée
        self._np_sigmoid = np.vectorize(self._sigmoid)

    @staticmethod
    def _sigmoid(value):
        return 1 / (1 + math.exp(-value))

    def initialize_network(self):
        # Initialisation des poids aléatoires
        weights_input = np.random.rand(self.SIZE_HIDDEN, self.SIZE_INPUT + 1) / self.SIZE_INPUT
        weights_hidden = np.random.rand(self.SIZE_OUTPUT, self.SIZE_HIDDEN + 1) / self.SIZE_HIDDEN

        self.weights = [weights_input, weights_hidden]

    def train(self, image, label):
        # Propagation de l'activité
        outputs_h = self._get_hidden_outputs(image)
        outputs = self._get_outputs_from_hidden(outputs_h)

        input_data = np.append(image, 1)

        # Calcul des erreurs des sorties
        errors_output = label - outputs

        # Calcul des erreurs de la couche cachée
        errors_hidden = outputs_h * (1 - outputs_h) * np.dot(errors_output, self.weights[1])

        # Correction des poids couche cachée
        for i in range(self.SIZE_HIDDEN):
            self.weights[0][i] += self.TRAIN_STEP_HIDDEN * errors_hidden[i] * input_data

        # Correction des poids couche de sortie
        for i in range(self.SIZE_OUTPUT):
            self.weights[1][i] += self.TRAIN_STEP_OUTPUT * errors_output[i] * outputs_h

    def _get_hidden_outputs(self, inputs):
        # Ajout de l'entrée virtuelle à 1.0 pour la couche cachée
        input_data = np.append(inputs, 1)
        # Calcul de l'activité de la couche cachée
        out = self._np_sigmoid(np.dot(self.weights[0][:], input_data))
        # Ajout de l'entrée virtuelle à 1.0 pour la couche de sortie
        return np.append(out, 1)

    def _get_outputs_from_hidden(self, outputs_hidden):
        # Calcul de l'activité pour la couche de sortie
        return np.dot(self.weights[1], outputs_hidden)

    def get_outputs(self, inputs):
        return self._get_outputs_from_hidden(self._get_hidden_outputs(inputs))

    def print_params(self):
        print(" • η hidden =", self.TRAIN_STEP_HIDDEN)
        print(" • η output =", self.TRAIN_STEP_OUTPUT)
        print(" • Hidden layer size :", self.SIZE_HIDDEN)


if __name__ == '__main__':
    shallow = Shallow()
    shallow.dataset_train()
    shallow.test()
