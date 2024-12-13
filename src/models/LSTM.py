import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTM(nn.Module):
    def __init__(self, hidden_size=10, num_layers=1, learning_rate=1e-3, device='cpu'):
        """
        Classe pour un modèle LSTM.

        Paramètres :
        - hidden_size (int) : Dimension de l'espace caché.
        - num_layers (int) : Nombre de couches LSTM.
        - learning_rate (float) : Taux d'apprentissage.
        - device (str) : Dispositif ('cpu' ou 'cuda').
        """
        super(LSTM, self).__init__()

        # Paramètres du modèle
        self.input_size = None
        self.output_size = None
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.device = device

        # Définir le modèle LSTM
        self.model = None
        self.fc = None

        # Définir la fonction de perte et l'optimiseur
        self.optimizer = None
        self.criterion = None

    def train(self, X, Y, epochs=100, batch_size=32, classification=False, prediction_timesteps=[]):
        """
        Entraîne le modèle LSTM.

        Paramètres :
        - X : Données d'entrée (numpy array ou tenseur). (sample, time, input_dim)
        - Y : Données de sortie (numpy array ou tenseur). (sample, time, output_dim)
        - epochs (int) : Nombre d'époques.
        - batch_size (int) : Taille des mini-lots.
        - classification (bool) : Indique si la tâche est une classification.
        - prediction_timesteps (list) : Liste des indices de temps pour lesquels prédire.
        """

        # Convertir les données en tenseurs PyTorch
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y, dtype=torch.float32, device=self.device)
        
        # Define model
        self._define_model(X.shape[-1], Y.shape[-1])
        if classification:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()  # Vous pouvez changer cela en fonction de votre tâche
        
        # Créer un DataLoader
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Entraîner le modèle
        self.model.train()
        for _ in range(epochs):
            for i, (batch_X, batch_Y) in enumerate(dataloader):
                # Initialiser les états cachés
                h0 = torch.zeros(self.num_layers, batch_X.size(0), self.hidden_size, device=self.device)
                c0 = torch.zeros(self.num_layers, batch_X.size(0), self.hidden_size, device=self.device)

                # Forward pass
                outputs, _ = self.model(batch_X, (h0, c0)) # outputs: (batch_size, seq_length, hidden_size)
                outputs = self.fc(outputs) # outputs: (batch_size, seq_length, output_size) 

                # Select only the prediction timesteps
                preds = []
                truths = []
                for j in range(batch_X.shape[0]):
                    sample = i*batch_size + j
                    preds += [outputs[j, prediction_timesteps[sample], :]]
                    truths += [batch_Y[j, prediction_timesteps[sample], :]]
                preds = torch.stack(preds)
                truths = torch.stack(truths)

                # Compute loss
                loss = self.criterion(preds, truths)

                # Backward pass et mise à jour des poids
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def run(self, X):
        """
        Génère des prédictions avec le modèle LSTM.
        
        Paramètres :
        - X : Tenseur d'entrée (batch_size, seq_len, input_dim).
        
        Retourne :
        - Tenseur de sortie (batch_size, seq_len, output_dim).
        """
        # Convertir les données en tenseurs PyTorch
        X = torch.tensor(X, dtype=torch.float32, device=self.device)

        # Initialiser les états cachés
        h0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, X.size(0), self.hidden_size, device=self.device)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model(X, (h0, c0))
            outputs = self.fc(outputs)

        return outputs.cpu().numpy()

    def count_params(self):
        """"
        Compte le nombre de paramètres du modèle.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _define_model(self, input_size, output_size):
        """
        Définit le modèle LSTM.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.model = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, device=self.device)
        self.fc = nn.Linear(self.hidden_size, output_size, device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    


