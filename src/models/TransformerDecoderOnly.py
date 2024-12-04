import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TransformerDecoderOnly(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1, learning_rate=1e-3, device='cpu'):
        """
        Classe pour un modèle Transformer utilisant uniquement le décodeur.

        Paramètres :
        - d_model (int) : Dimension du modèle.
        - nhead (int) : Nombre de têtes dans l'attention multi-têtes.
        - num_layers (int) : Nombre de couches du décodeur.
        - dim_feedforward (int) : Dimension de la couche feedforward.
        - dropout (float) : Taux de dropout.
        - learning_rate (float) : Taux d'apprentissage.
        - device (str) : Dispositif ('cpu' ou 'cuda').
        """
        super(TransformerDecoderOnly, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device
        self.input_size = None
        self.output_size = None

        # Modèle Decoder et FC
        self.fc_in = None
        self.transformer = None
        self.fc_out = None

        # Définir la fonction de perte et l'optimiseur
        self.optimizer = None
        self.criterion = None

    def train(self, X, Y, epochs=10, batch_size=32, classification=False, prediction_start=0):
        """
        Entraîne le modèle TransformerDecoderOnly.

        Paramètres :
        - X : Données d'entrée (numpy array ou tenseur). (sample, time, input_dim)
        - Y : Données de sortie (numpy array ou tenseur). (sample, time, output_dim)
        - epochs (int) : Nombre d'époques.
        - batch_size (int) : Taille des mini-lots.
        - classification (bool) : Indique si la tâche est une classification.
        - prediction_start (int) : Indice de début de prédiction.
        """
        # Convertir les données en tenseurs PyTorch
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y, dtype=torch.float32, device=self.device)

        # Définir le modèle
        self._define_model(X.shape[-1], Y.shape[-1], classification=classification)

        # Créer un DataLoader
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Générer le masque de séquence
        mask_seq = self._generate_sequence_mask(Y.shape[1])

        # Entraîner le modèle
        self.transformer.train()

        for _ in range(epochs):
            for X_batch, Y_batch in dataloader:
                # Forward pass
                emb_X = self.fc_in(X_batch)  # Embedding de l'entrée
                tr_output = self.transformer(src=emb_X, mask=mask_seq)  # Transformer
                output = self.fc_out(tr_output)  # Projection finale

                # Calculer la perte
                loss = self.criterion(output[:, prediction_start:, :], Y_batch[:, prediction_start:, :])

                # Backward pass et mise à jour des poids
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def run(self, X):
        """
        Génère des prédictions avec le modèle TransformerDecoderOnly.

        Paramètres :
        - X : Tenseur d'entrée (batch_size, seq_len, input_dim).

        Retourne :
        - torch.Tensor : Séquence prédite (batch_size, seq_len, output_dim).
        """
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        mask_seq = self._generate_sequence_mask(X.shape[1])

        self.transformer.eval()
        with torch.no_grad():
            emb_X = self.fc_in(X)
            tr_output = self.transformer(src=emb_X, mask=mask_seq)
            output = self.fc_out(tr_output)
        return output

    def count_params(self):
        """
        Compte le nombre de paramètres du modèle.

        Retourne :
        - int : Le nombre total de paramètres entraînables.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _define_model(self, input_size, output_size, classification=False):
        """
        Définir le modèle TransformerDecoderOnly.

        Paramètres :
        - input_size (int) : Dimension de l'entrée.
        - output_size (int) : Dimension de la sortie.
        """
        self.input_size = input_size
        self.output_size = output_size

        # Définir les couches
        self.fc_in = nn.Linear(input_size, self.d_model, device=self.device)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                device=self.device,
                batch_first=True,
            ),
            num_layers=self.num_layers,
        )
        self.fc_out = nn.Linear(self.d_model, output_size, device=self.device)

        # Définir la fonction de perte et l'optimiseur
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        if classification:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

    def _generate_sequence_mask(self, length):
        """
        Génère un masque de séquence pour le décodeur Transformer.

        Paramètres :
        - length (int) : Longueur de la séquence.

        Retourne :
        - torch.Tensor : Masque de séquence (causal).
        """
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)
