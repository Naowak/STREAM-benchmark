import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Transformers(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, dropout=0.1, learning_rate=1e-3, device='cpu'):
        """
        Classe pour un modèle Transformer configurable.

        Paramètres :
        - d_model (int) : Dimension du modèle.
        - nhead (int) : Nombre de têtes dans l'attention multi-têtes.
        - num_layers (int) : Nombre de couches d'encodeurs.
        - dim_feedforward (int) : Dimension de la couche feedforward.
        - dropout (float) : Taux de dropout.
        - learning_rate (float) : Taux d'apprentissage.
        - device (str) : Dispositif ('cpu' ou 'cuda').
        """
        super(Transformers, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device
        self.input_size = None
        self.output_size = None

        # Modèle Transformer et FC
        self.fc_in_encoder = None
        self.dc_in_decoder = None
        self.transformer = None
        self.fc_out = None

        # Définir la fonction de perte et l'optimiseur
        self.optimizer = None
        self.criterion = None

    def train(self, X, Y, epochs=10, batch_size=32, classification=False):
        """
        Entraîne le modèle Transformer.

        Paramètres :
        - X : Données d'entrée (numpy array ou tenseur).
        - Y : Données de sortie (numpy array ou tenseur).
        - epochs (int) : Nombre d'époques.
        - batch_size (int) : Taille des mini-lots.
        - classification (bool) : Indique si la tâche est une classification.
        """
        # Convertir les données en tenseurs PyTorch
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y, dtype=torch.float32, device=self.device)
        
        # Définir le modèle
        self._define_model(X.shape[-1], Y.shape[-1], classification=classification)

        # Créer un DataLoader
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Générer les masques de séquence
        mask_seq_enc = self._generate_sequence_mask(X.shape[1])
        mask_seq_dec = self._generate_sequence_mask(Y.shape[1], mask_current=True)

        # Entraîner le modèle
        self.transformer.train()

        # Entraîner le modèle
        for _ in range(epochs):
            for X_batch, Y_batch in dataloader:
                # Forward pass
                emb_X = self.fc_in_encoder(X_batch)
                emb_Y = self.fc_in_decoder(Y_batch)       
                tr_output = self.transformer(src=emb_X, tgt=emb_Y, src_mask=mask_seq_enc, tgt_mask=mask_seq_dec)
                output = self.fc_out(tr_output)

                # Calculer la perte
                loss = self.criterion(output, Y_batch)

                # Backward pass et mise à jour des poids
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def run(self, X):
        """
        Génère des prédictions avec le modèle Transformer.

        Paramètres :
        - X : Tenseur d'entrée (batch_size, seq_len, input_dim).

        Retourne :
        - torch.Tensor : Séquence prédite (batch_size, seq_len, output_dim).
        """
        # Convertir l'entrée en tenseur PyTorch
        X = torch.tensor(X, dtype=torch.float32, device=self.device)

        # Créer un DataLoader
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Générer les masques de séquence
        mask_memory = self._generate_memory_mask(X.shape[1], X.shape[1])
        mask_seq_enc = self._generate_sequence_mask(X.shape[1])
        mask_seq_dec = self._generate_sequence_mask(X.shape[1], mask_current=True)

        # Run model
        Y_preds = []
        self.transformer.eval()
        with torch.no_grad():
            # Iterate over batches
            for [X_batch] in dataloader:

                # Init Y_batch and compute emb_X for the encoder
                Y_batch = torch.zeros((X_batch.shape[0], X_batch.shape[1], self.output_size), device=self.device)
                emb_X = self.fc_in_encoder(X_batch)

                # Iterate over the sequence to predict the next Y step by step
                for i in range(X_batch.shape[1]):
                    emb_Y = self.fc_in_decoder(Y_batch)
                    tr_output = self.transformer(src=emb_X, tgt=emb_Y, src_mask=mask_seq_enc, tgt_mask=mask_seq_dec, memory_mask=mask_memory)
                    Y_batch[:, i, :] = self.fc_out(tr_output[:, i, :])
                
                # Append the predictions
                Y_preds.append(Y_batch)
            
        return torch.cat(Y_preds, dim=0)

    def count_params(self):
        """
        Compte le nombre de paramètres du modèle.

        Retourne :
        - int : Le nombre total de paramètres entraînables.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _define_model(self, input_size, output_size, classification=False):
        """
        Définir le modèle Transformer.

        Paramètres :
        - input_size (int) : Dimension de l'entrée.
        - output_size (int) : Dimension de la sortie.
        """
        # Define input and output sizes
        self.input_size = input_size
        self.output_size = output_size

        # Define the layers
        self.fc_in_encoder = nn.Linear(input_size, self.d_model, device=self.device)
        self.fc_in_decoder = nn.Linear(output_size, self.d_model, device=self.device)
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
            device=self.device
        )
        self.fc_out = nn.Linear(self.d_model, output_size, device=self.device)

        # Define the loss function and optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        if classification:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

    def _generate_sequence_mask(self, length, mask_current=False):
        """
        Génère un masque de séquence pour le modèle Transformer.

        Paramètres :
        - length (int) : Taille de la séquence.
        - see_current (bool) : Indique si le modèle peut voir l'élement actuel.

        Retourne :
        - torch.Tensor : Masque de séquence.
        """
        if mask_current:
            length += 1
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask if not mask_current else mask[:-1, 1:]
    
    def _generate_memory_mask(self, src_len, tgt_len):
        """
        Génère un masque de mémoire pour le modèle Transformer.

        Paramètres :
        - src_len (int) : Taille de la séquence source.
        - tgt_len (int) : Taille de la séquence cible.

        Retourne :
        - torch.Tensor : Masque de mémoire.
        """
        mask = torch.ones(tgt_len, src_len)
        for i in range(tgt_len):
            mask[i, i+1:] = float('-inf')
        return mask


