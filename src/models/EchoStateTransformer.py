import torch
from tqdm import tqdm

class EchoStateTransformer(torch.nn.Module):
    """Implementation of the Echo State Transformer model."""

    def __init__(self, memory_units=4, memory_dim=100, attention_dim=3, learning_rate=1e-3, weight_decay=1e-5):
        """
        Initialize the Echo State Transformer.

        Parameters:
        - M (int): Number of memory units
        - R (int): Neurons per memory unit
        - D (int): Input dimension per unit
        - O (int): Output dimension
        - leak_rate (float): Leak rate for state update
        """
        super(EchoStateTransformer, self).__init__()
        self.memory_units = memory_units # M
        self.memory_dim = memory_dim # R
        self.attention_dim = attention_dim # D
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.input_dim = None # I
        self.output_dim = None # O

        # Initialize memory with fixed weights
        self.memory = None
        
        # Query, Key, Value weights [M, I, D]
        self.Wq = None
        self.Wk = None
        self.Wv = None

        # Output weight
        self.Wout = None

        # Optimizer & Loss
        self.optimizer = None
        self.criterion = None


    def forward(self, Xi, Si=None):
        """
        Forward pass of the Echo State Transformer.

        Parameters:
        - Xi (torch.Tensor): Input tensor [B, I]
        - Si (torch.Tensor, optional): Initial states [B, M, R]. Defaults to None.

        Returns:
        - Yi (torch.Tensor): Output tensor [B, O]
        """
        # /!\ Obviously there is a choice to make the memory initialization
        if Si is None:
            # Initialize states if not provided
            Si = torch.ones(Xi.shape[0], self.memory.units, self.memory.neurons, device=Xi.device) # [B, M, R]
        
        # Init input & previous state
        Xi = Xi # [B, I]
        Si_ = Si # [B, M, R]

        # Compute queries, keys, values
        Qi = (Xi @ self.Wq).transpose(0, 1).unsqueeze(1) # [B, 1, M, D]
        Ki_ = Si_.unsqueeze(1) @ self.Wk.unsqueeze(0) # [B, M, M, D]
        Vi_ = Si_.unsqueeze(1) @ self.Wv.unsqueeze(0) # [B, M, M, D]

        # Compute attention scores and update
        Ai = torch.nn.functional.softmax(Qi @ Ki_.transpose(-1, -2), dim=-1) / (self.memory.input_dim ** 0.5) # [B, M, M, M]
        Ui = Ai @ Vi_ # [B, M, M, D]
        Ui = Ui.view(-1, self.memory.units, 1, self.memory.input_dim) # [B, M, 1, M*D]

        # State update computation
        Si = self.memory(Ui, Si_) # [B, M, R]
        Si_flat = Si.view(-1, self.memory.units * self.memory.neurons) # [B, M*R]

        # Compute final output 
        Yi = Si_flat @ self.Wout # [B, O]

        return Yi, Si # [B, O], [B, M, R]

    def train(self, X, Y, epochs=100, batch_size=32, classification=False, prediction_timesteps=[]):
        """
        Training function for the Echo State Transformer model.

        Parameters:
            - X (torch.Tensor): Input tensor [sample, sequence, dim]
            - Y (torch.Tensor): Target tensor [sample, sequence, dim]
            - batch_size (int): Number of samples per batch

        Returns:
            - total_loss (float): Total loss over all batches
        """
        # Convert input and target tensors to float32
        X = torch.tensor(X).float()
        Y = torch.tensor(Y).float()

        # Define the model
        self._define_model(X.shape[-1], Y.shape[-1], classification)

        # Calcul du nombre de batches
        num_batches = X.shape[0] // batch_size
        loss_history = []

        # Train the model
        for epoch in range(epochs):
            tqdm_bar = tqdm(range(num_batches), desc='Training')
            for batch_idx in tqdm_bar:
                # Extraction d'un batch de données
                X_batch = X[batch_idx * batch_size : (batch_idx + 1) * batch_size] # [B, T, I]
                Y_batch = Y[batch_idx * batch_size : (batch_idx + 1) * batch_size] # [B, T, O]

                # Initialiser les états cachés pour la séquence
                states = None
                outputs = []

                # Forward pass pour chaque temps
                for t in range(X_batch.shape[1]):
                    y_out, states = self.forward(X_batch[:, t], states) # [B, O], [B, M, R]
                    outputs.append(y_out)
                outputs = torch.stack(outputs, dim=1) # [B, T, O]

                # Select only the prediction timesteps
                preds = []
                truths = []
                for j in range(X_batch.shape[0]):
                    sample = batch_idx * batch_size + j
                    preds += [outputs[j, prediction_timesteps[sample], :]]
                    truths += [Y_batch[j, prediction_timesteps[sample], :]]
                preds = torch.stack(preds, dim=0) # [B, prediction_timesteps, O]
                truths = torch.stack(truths, dim=0) # [B, prediction_timesteps, O]
                
                # Calcul du loss
                loss = self.criterion(preds, truths)
                loss_history.append(loss.item())

                # Backward pass et mise à jour des paramètres
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tqdm_bar.set_description(f"Epoch: {epoch+1}/{epochs} - Batch: {batch_idx+1}/{num_batches} - Loss: {loss:.4f}")
        
        return loss_history
    
    def run(self, X, batch_size=10):
        """
        Run the Echo State Transformer model on the input tensor.
        
        Parameters:
            - X (torch.Tensor): Input tensor [sample, sequence, dim]
            - batch_size (int): Number of samples per batch
        
        Returns:
            - Y_hat (torch.Tensor): Predicted tensor [sample, sequence, dim]
        """
        # Convert input tensor to float32
        X = torch.tensor(X).float()

        # Calcul du nombre de batches
        num_batches = X.shape[0] // batch_size
        Y_hat = []

        tqdm_bar = tqdm(range(num_batches), desc='Prediction')

        with torch.no_grad():
            for batch_idx in tqdm_bar:
                # Extraction d'un batch de données
                X_batch = X[batch_idx * batch_size : (batch_idx + 1) * batch_size]

                # Initialiser les états cachés pour la séquence
                states = None
                outputs = []

                # Forward pass pour chaque temps
                for t in range(X_batch.shape[1]):
                    with torch.no_grad():
                        y_out, states = self.forward(X_batch[:, t], states) # [B, O], [B, M, R]
                        outputs.append(y_out)
                
                # Convertir la séquence de sorties en tenseur [batch_size, sequence_length, output_dim]
                Y_hat_batch = torch.stack(outputs, dim=1)
                Y_hat.append(Y_hat_batch)
            
            # Concaténer les batches de prédictions
            Y_hat = torch.cat(Y_hat, dim=0)

        return Y_hat.cpu().numpy()

    def count_params(self):
        """
        Count the number of parameters in the network.

        Returns:
        - num_params (int): Number of parameters in the network
        """
        return sum(p.numel() for p in self.parameters())

    def _define_model(self, input_dim, output_dim, classification=False):
        """
        Define the model architecture.

        Parameters:
        - input_dim (int): Input dimension.
        - output_dim (int): Output dimension.
        - classification (bool): Indicates if the task is a classification task.
        """
        # Define the input and output dims
        self.input_dim = input_dim # I
        self.output_dim = output_dim # O

        # Initialize memory with fixed weights
        self.memory = Memory(units=self.memory_units, neurons=self.memory_dim, input_dim=self.memory_units * self.attention_dim)

        # Query, Key, Value weights
        self.Wq = torch.nn.Parameter(torch.randn(self.memory_units, self.input_dim, self.attention_dim)) # [M, I, D]
        self.Wk = torch.nn.Parameter(torch.randn(self.memory_units, self.memory_dim, self.attention_dim)) # [M, R, D]
        self.Wv = torch.nn.Parameter(torch.randn(self.memory_units, self.memory_dim, self.attention_dim)) # [M, R, D]

        # Output weight
        self.Wout = torch.nn.Parameter(torch.randn(self.memory_units * self.memory_dim, self.output_dim)) # [M*R, O]

        # Initialiser l'optimiseur
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if classification:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()







class Memory(torch.nn.Module):
    """Implements a reservoir network."""

    def __init__(self, units=None, neurons=None, input_dim=None, sr=0.5, lr=0.4, input_scaling=1.0, rc_connectivity=0.2, input_connectivity=0.2, bias_prob=0.5):
        """
        Create a reservoir with the given parameters.

        Parameters:
        - units (int): Number of reservoirs.
        - neurons (int): Number of neurons in each reservoir.
        - input_dim (int): Input dimension.
        - sr (float): Spectral radius of the recurrent weight matrix.
        - input_scaling (float): Input scaling.
        - rc_connectivity (float): Connectivity of the recurrent weight matrix.
        - input_connectivity (float): Connectivity of the input weight matrix.
        """
        super(Memory, self).__init__()
        # Check the parameters
        if units is None or neurons is None or input_dim is None:
            raise ValueError("You must provide the number of units, neurons and input dimension")
        
        # Store the parameters
        self.units = units # M
        self.neurons = neurons # R
        self.input_dim = input_dim # M*D
        self.sr = torch.nn.Parameter(torch.rand(units, 1, 1))
        self.lr = torch.nn.Parameter(torch.rand(units, 1))
        self.input_scaling = input_scaling
        self.rc_connectivity = rc_connectivity
        self.input_connectivity = input_connectivity
        self.bias_prob = bias_prob

        # Initialize the recurrent weight matrix W (normal distribution)
        Ws = []
        Wins = []
        biases = []
        for i in range(units):
            # Initialize the weight matrices
            W = _initialize_matrix((neurons, neurons), rc_connectivity, distribution='normal')
            Win = _initialize_matrix((input_dim, neurons), input_connectivity, distribution='bernoulli')
            bias = _initialize_matrix((1, neurons), bias_prob, distribution='bernoulli')

            # Append the initialized matrices to the lists
            Ws.append(W)
            Wins.append(Win)
            biases.append(bias)
        
        # Convert the lists to tensors and register them as buffers with requires_grad=False
        self.register_buffer('W', torch.stack(Ws, dim=0)) # [M, R, R]
        self.register_buffer('Win', torch.stack(Wins, dim=0)) # [M, D, R]
        self.register_buffer('bias', torch.stack(biases, dim=0)) # [M, 1, R]
    
    def forward(self, X, state=None):
        """
        Forward pass of the reservoir network.
        
        Parameters:
        - X (torch.Tensor): Input tensor [batch, units, input_dim].
        - state (torch.Tensor, optional): Initial states [batch, units, neurons]. Defaults to None.
        """
        if state is None:
            raise ValueError("You must provide the initial states to the memory")
        
        # Retrieve the batch size
        batch_size = X.shape[0] # B
        
        # Feed
        feed = X.view(batch_size, self.units, 1, self.input_dim) @ self.Win # [B, M, 1, R]

        # Echo
        W = _set_spectral_radius(self.W, self.sr)
        echo = state.view(batch_size, self.units, 1, self.neurons) @ W + self.bias # [B, M, 1, R]

        # New state
        new_state = (1 - self.lr) * state + self.lr * torch.tanh(feed + echo).squeeze() # [B, M, R]
        return new_state
    


def _initialize_matrix(shape, connectivity, distribution='normal', **kwargs):
    """
    Initialize a matrix with a given shape and connectivity.

    Parameters:
    - shape (tuple): Shape of the matrix.
    - connectivity (float): Connectivity of the matrix.
    - distribution (str): Distribution of the matrix values ('normal' or 'bernoulli').
    - kwargs: Additional arguments for the distribution.

    Returns:
    - torch.Tensor: Initialized matrix.
    """
    if distribution == 'normal':
        matrix = torch.randn(shape)
        mask = torch.rand(shape) < connectivity
        return matrix * mask
    
    elif distribution == 'bernoulli':
        return torch.bernoulli(torch.full(shape, connectivity))
    
    else:
        raise ValueError("Unsupported distribution type")

def _set_spectral_radius(matrix, sr):
    """
    Set the spectral radius of a matrix.

    Parameters:
    - matrix (torch.Tensor): The matrix to adjust.
    - sr (float): The desired spectral radius.

    Returns:
    - torch.Tensor: The matrix with the adjusted spectral radius.
    """
    eigenvalues = torch.linalg.eigvals(matrix)
    max_eigenvalue = torch.max(torch.abs(eigenvalues), dim=-1).values.reshape(-1, 1, 1)
    return matrix * (sr / max_eigenvalue)

















