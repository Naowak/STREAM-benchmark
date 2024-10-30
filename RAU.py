from sklearn.linear_model import Ridge
import numpy as np


class RAU:

    def __init__(self, input_dim, output_dim, units=100, degree=3, spectral_radius=1, leak_rate=0.5, reservoir_kind='single_attention'):
        """
        Initialise le réseau de neurones récurrents.

        Paramètres :
        input_dim (int) : La dimension de l'entrée.
        output_dim (int) : La dimension de la sortie.
        units (int) : Le nombre d'unités dans le réservoir.
        degree (int) : Le nombre de connexions entrantes par unité.
        spectral_radius (float) : Le rayon spectral de la matrice de poids du réservoir.
        leak_rate (float) : Le taux de fuite.
        reservoir_kind (str) : Le type de réservoir (single_attention, multiple_attention, no_attention).
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.units = units
        self.degree = degree
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate
        self.reservoir_kind = reservoir_kind

        self.Win, self.W, self.bias = self._init_weights(input_dim, units, degree, spectral_radius)
        self.readout = None
        
    def _init_weights(self, input_dim, units=100, degree=3, spectral_radius=1):
        """
        Définit les poids du réseau de neurones récurrents.

        Paramètres :
        input_dim (int) : La dimension de l'entrée.
        units (int) : Le nombre d'unités dans le réservoir.
        degree (int) : Le nombre de connexions entrantes par unité.

        Retourne :
        tuple : Un tuple contenant trois np.ndarray, Win, W et bias, qui sont les poids de l'entrée, du réservoir et du biais respectivement.
        """
        # Win : connexion entre l'entrée et les unités du réservoir
        Win = np.random.binomial(1, 1, (units, input_dim)) # /!\ p=1 pour avoir des 1 partout

        # W : connexion entre les unités du réservoir
        W_mask = np.zeros((units, units))
        for i in range(units):
            indices = np.random.choice(units, degree, replace=False)
            W_mask[i, indices] = 1
        W = np.random.normal(0, 1, (units, units)) * W_mask
        max_eigenvalue = np.max(np.abs(np.linalg.eigvals(W)))
        W = W / max_eigenvalue
        W = W * spectral_radius

        # Bias : biais des unités du réservoir
        bias = np.random.normal(0, 1, (units, 1))

        return Win, W, bias

    def _compute_activity(self, X):
        """
        Calcule l'activité du réseau de neurones récurrents.

        Paramètres :
        X (np.ndarray) : Les données d'entrée.

        Retourne :
        np.ndarray : L'activité du réseau.
        """
        x = np.zeros((self.units, 1)) + self.bias
        activity = []
        for i in range(len(X)):
            queries = self.Win @ X[i].reshape(-1, 1)
            keys = self.W @ x
            if self.reservoir_kind == 'single_attention':
                x = x * (1 - self.leak_rate) + np.tanh(np.sum(queries * keys, axis=1).reshape(-1, 1) + self.bias) * self.leak_rate
            elif self.reservoir_kind == 'multiple_attention':
                x = x * (1 - self.leak_rate) + np.tanh(np.sum(queries @ keys.T).reshape(-1, 1) + self.bias) * self.leak_rate
                #x = x * (1 - self.leak_rate) + np.tanh(np.max(queries @ keys.T).reshape(-1, 1) + self.bias) * self.leak_rate
            elif self.reservoir_kind == 'no_attention':
                inputs = np.concatenate([queries, keys], axis=1)
                x = x * (1 - self.leak_rate) + np.tanh(np.sum(inputs, axis=1).reshape(-1, 1) + self.bias) * self.leak_rate
            else:
                raise ValueError('Invalid reservoir kind')
            activity.append(x)
        activity = np.array(activity)

        return activity
    
    def train(self, X, Y):
        # Compute activity for X with attention units
        activity = self._compute_activity(X)

        # Train the readout
        self.readout = Ridge(alpha=1e-7)
        self.readout.fit(activity.reshape(-1, self.units), Y.reshape(-1, self.output_dim))

    def run(self, X):
        activity = self._compute_activity(X)
        return self.readout.predict(activity.reshape(-1, self.units))






