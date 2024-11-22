import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTM(nn.Module):
    def __init__(self, hidden_size=10, num_layers=1, learning_rate=1e-3, device='cpu'):
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

    def train(self, X, Y, epochs=100, batch_size=32, classification=False):

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
            for batch_X, batch_Y in dataloader:
                # Initialiser les états cachés
                h0 = torch.zeros(self.num_layers, batch_X.size(0), self.hidden_size, device=self.device)
                c0 = torch.zeros(self.num_layers, batch_X.size(0), self.hidden_size, device=self.device)

                # Forward pass
                outputs, _ = self.model(batch_X, (h0, c0)) # outputs: (batch_size, seq_length, hidden_size)
                outputs = self.fc(outputs) # outputs: (batch_size, seq_length, output_size) 

                # Calculer la perte
                loss = self.criterion(outputs, batch_Y)

                # Backward pass et mise à jour des poids
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def run(self, X):
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
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _define_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, device=self.device)
        self.fc = nn.Linear(self.hidden_size, output_size, device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    


# # Exemple d'utilisation
# input_size = 10  # Exemple de taille d'entrée
# hidden_size = 20  # Exemple de taille cachée
# output_size = 1  # Exemple de taille de sortie

# # Créer une instance du modèle
# lstm_model = LSTMModel(input_size, hidden_size, output_size)

# # Données d'exemple
# X_train = np.random.randn(100, 50, input_size)  # 100 séquences de longueur 50
# Y_train = np.random.randn(100, output_size)  # 100 cibles

# # Entraîner le modèle
# lstm_model.train(X_train, Y_train, epochs=10, batch_size=16)

# # Exécuter le modèle
# X_test = np.random.randn(10, 50, input_size)  # 10 séquences de longueur 50
# predictions = lstm_model.run(X_test)

# # Compter les paramètres
# num_params = lstm_model.count_params()
# print(f'Nombre de paramètres: {num_params}')
