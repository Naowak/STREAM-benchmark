import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Transformer(nn.Module):
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
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device

        # Modèle Transformer et FC
        self.fc_in_encoder = None
        self.dc_in_decoder = None
        self.transformer = None
        self.fc_out = None

        # Définir la fonction de perte et l'optimiseur
        self.optimizer = None
        self.criterion = None

    def _define_model(self, input_size, output_size):
        """
        Définir le modèle Transformer.

        Paramètres :
        - input_size (int) : Dimension de l'entrée.
        - output_size (int) : Dimension de la sortie.
        """
        self.fc_in_encoder = nn.Linear(input_size, self.d_model, device=self.device)
        self.fc_in_decoder = nn.Linear(input_size, self.d_model, device=self.device)
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
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)


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
        self._define_model(X.shape[-1], Y.shape[-1])
        if classification:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()

        # Créer un DataLoader
        dataset = TensorDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Entraîner le modèle
        self.transformer.train()
        # for _ in range(epochs):
        #     for batch_X, batch_Y in dataloader:
        #         # Forward pass
        #         outputs = self._forward(batch_X)

        #         # Calculer la perte
        #         loss = self.criterion(outputs, batch_Y)

        #         # Backward pass et mise à jour des poids
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         self.optimizer.step()

    # def run(self, X, max_len=50, start_token=None):
    #     """
    #     Generate predictions with the Transformer model.

    #     Parameters:
    #     - X: Input tensor (batch_size, seq_len_src, input_dim).
    #     - max_len: Maximum sequence length to generate.
    #     - start_token: Initial token for target sequence (e.g., start token).

    #     Returns:
    #     - torch.Tensor: Predicted sequence (batch_size, seq_len_tgt, output_dim).
    #     """
    #     # Convert input to PyTorch tensor
    #     X = torch.tensor(X, dtype=torch.float32, device=self.device)

    #     # Encode the source sequence
    #     memory = self._forward(X, Y=None)

    #     # Initialize target sequence with start token
    #     batch_size = X.size(0)
    #     if start_token is None:
    #         start_token = torch.zeros(batch_size, 1, self.d_model, device=self.device)

    #     Y = start_token
    #     for _ in range(max_len):
    #         # Generate mask for auto-regressive decoding
    #         tgt_mask = nn.Transformer.generate_square_subsequent_mask(Y.size(1)).to(self.device)

    #         # Decode step-by-step
    #         output = self.transformer.decoder(
    #             Y, memory, tgt_mask=tgt_mask, memory_key_padding_mask=(X.sum(dim=-1) == 0)
    #         )
    #         next_token = self.fc(output[:, -1:, :])  # Predict next token

    #         # Append the new token to the target sequence
    #         Y = torch.cat([Y, next_token], dim=1)

    #         # Optional stopping condition (e.g., end token) can be added here

    #     return Y

    # def count_params(self):
    #     """
    #     Compte le nombre de paramètres du modèle.

    #     Retourne :
    #     - int : Le nombre total de paramètres entraînables.
    #     """
    #     return sum(p.numel() for p in self.parameters() if p.requires_grad)



    # def _forward(self, X, Y=None):
    #     """
    #     Perform a forward pass with the Transformer model.

    #     Parameters:
    #     - X: Source sequence (batch_size, seq_len_src, input_dim).
    #     - Y: Target sequence (batch_size, seq_len_tgt, output_dim).

    #     Returns:
    #     - torch.Tensor: Output from the model (batch_size, seq_len_tgt, output_dim).
    #     """
    #     # Masks
    #     src_mask = None
    #     tgt_mask = None
    #     memory_mask = None

    #     # Padding masks for sequences
    #     src_key_padding_mask = (X.sum(dim=-1) == 0)  # Identify padding (batch_size, seq_len_src)
    #     tgt_key_padding_mask = None if Y is None else (Y.sum(dim=-1) == 0)

    #     # Encoder: Process the source sequence
    #     memory = self.transformer.encoder(
    #         X,
    #         mask=src_mask,
    #         src_key_padding_mask=src_key_padding_mask
    #     )

    #     # Decoder: Use target sequence if provided
    #     if Y is not None:
    #         outputs = self.transformer.decoder(
    #             Y,
    #             memory,
    #             tgt_mask=tgt_mask,
    #             memory_mask=memory_mask,
    #             tgt_key_padding_mask=tgt_key_padding_mask,
    #             memory_key_padding_mask=src_key_padding_mask
    #         )
    #         return self.fc(outputs)
        
    #     # If no `Y`, return encoder output for future use
    #     return memory

