
from network import Network

import torch
import torch.nn as nn


class Deep(Network):

    def __init__(self):
        super().__init__()

        # Taux d'apprentissage
        self.TRAIN_STEP = 0.05

        # Nombre de couches cachées
        self.NUMBER_HIDDEN = 2
        # Taille des couches cachées
        self.SIZE_HIDDEN = 50

        # Le modèle des couches d'apprentissage
        self.model = None

        # La fonction loss pour le calcul de gradient
        self.loss_fn = torch.nn.MSELoss(reduction='sum')

    class PerceptronLayer(nn.Module):

        def __init__(self, input_size, output_size):
            super().__init__()

            # Une couche cachée du modèle perceptron multi-couche est composée
            # d'une couche linéaire et d'une couche appliquant la fonction sigmoide
            self.linear_layer = nn.Linear(input_size, output_size)
            self.sigmoid_layer = nn.Sigmoid()

        def forward(self, input):
            out = self.linear_layer(input)
            out = self.sigmoid_layer(out)
            return out

    def initialize_network(self):
        # Initialisation de NUMBER_HIDDEN couches cachées
        hidden_layers = [self.PerceptronLayer(self.SIZE_HIDDEN, self.SIZE_HIDDEN)] * self.NUMBER_HIDDEN

        self.model = torch.nn.Sequential(
            self.PerceptronLayer(self.SIZE_INPUT, self.SIZE_HIDDEN),
            *hidden_layers,  # "Unwrap" les couches cachées en paramètres
            torch.nn.Linear(self.SIZE_HIDDEN, self.SIZE_OUTPUT),
        )

        # On définie les poids comme ayant été initialisés
        self.weights = True

    def train(self, image, label):
        input_label = torch.from_numpy(label)
        outputs = self.get_outputs(image)

        loss = self.loss_fn(outputs, input_label)
        # Remise a 0 des gradients
        self.model.zero_grad()

        # Calcul du gradient
        loss.backward()

        # Apprentissage
        with torch.no_grad():
            for param in self.model.parameters():
                param.data -= self.TRAIN_STEP * param.grad

    def get_outputs(self, inputs):
        return self.model(torch.Tensor(inputs))

    def print_params(self):
        print(" • η =", self.TRAIN_STEP)
        print(" • Hidden layers =", self.NUMBER_HIDDEN)
        print(" • Hidden layers size =", self.SIZE_HIDDEN)


if __name__ == '__main__':
    shallow = Deep()
    shallow.dataset_train()
    shallow.test()
