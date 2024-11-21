from reservoirpy.nodes import Reservoir, Ridge, ESN

class ReservoirComputing():

    def __init__(self, n_units, spectral_radius, leak_rate, ridge):
        """
        Initialise le modèle de Reservoir Computing.
        
        Paramètres :
        n_units (int) : Le nombre d'unités dans le réservoir.
        spectral_radius (float) : Le rayon spectral de la matrice de poids du réservoir.
        leak_rate (float) : Le taux de fuite.
        ridge (float) : Le coefficient de régularisation Ridge.
        """
        self.reservoir = Reservoir(n_units, sr=spectral_radius, lr=leak_rate)
        self.readout = Ridge(ridge=ridge)
        self.esn = ESN(reservoir=self.reservoir, readout=self.readout, workers=-1)
    
    def train(self, X, Y):
        """
        Entraîne l'ESN sur les données d'entraînement.
        
        Paramètres :
        X (np.ndarray) : Les données d'entrée. (sample, time, input_dim) or (time, input_dim)
        Y (np.ndarray) : Les données de sortie. (sample, time, output_dim) or (time, output_dim)
        """
        self.esn.fit(X, Y)
    
    def run(self, X):
        """
        Prédit les données de sortie à partir des données d'entrée.

        Paramètres :
        X (np.ndarray) : Les données d'entrée. (sample, time, input_dim) or (time, input_dim)

        Retourne :
        np.ndarray : Les prédictions. (sample, time, output_dim)
        """
        return self.esn.run(X)

    def count_params(self):
        """
        Compte le nombre de paramètres du modèle.

        Retourne :
        int : Le nombre de paramètres du modèle.
        """
        params = 0
        if self.esn.reservoir.W is None:
            return params
        
        params += self.esn.reservoir.W_in.size
        params += self.esn.reservoir.W.size
        params += self.esn.reservoir.bias.size
        params += self.esn.readout.Wout.size
        params += self.esn.readout.bias.size
        return params
