from reservoirpy.nodes import Reservoir, Ridge
import reservoirpy as rpy
import numpy as np
rpy.verbosity(0)  # no need to be too verbose here

class ESN():

    def __init__(self, n_units, spectral_radius, leak_rate):
        """
        Initialise le modèle de Reservoir Computing.
        
        Paramètres :
        n_units (int) : Le nombre d'unités dans le réservoir.
        spectral_radius (float) : Le rayon spectral de la matrice de poids du réservoir.
        leak_rate (float) : Le taux de fuite.
        """
        # Hyperparamètres
        self.n_units = n_units
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate

        # ESN weights
        self.reservoir = Reservoir(n_units, sr=spectral_radius, lr=leak_rate)
        self.ridges = [Ridge(ridge=0.1**i) for i in range(1, 11)]
        self.readout = None
        self.model = None
    
    def train(self, X, Y):
        """
        Entraîne l'ESN sur les données d'entraînement.
        
        Paramètres :
        X (np.ndarray) : Les données d'entrée. (sample, time, input_dim) 
        Y (np.ndarray) : Les données de sortie. (sample, time, output_dim) 
        """
        # Make numpy arrays
        X = np.array(X)
        Y = np.array(Y)

        # Run reservoir
        states = []
        for i in range(X.shape[0]):
            states.append(self.reservoir.run(X[i]))
        states = np.array(states)

        # Train ridges
        errors = []
        for ridge in self.ridges:
            # Train ridge
            ridge.fit(states, Y)

            # Make prediction on train dataset
            y_preds = []
            for i in range(X.shape[0]):
                y_preds += [ridge.run(states[i])]
            y_preds = np.array(y_preds)

            # Compute errors on train dataset
            errors += [np.mean((y_preds - Y[i]) ** 2)]
        
        # Create best model
        self.readout = self.ridges[np.argmin(errors)]
        self.model = self.reservoir >> self.readout
    
    def run(self, X):
        """
        Prédit les données de sortie à partir des données d'entrée.

        Paramètres :
        X (np.ndarray) : Les données d'entrée. (sample, time, input_dim)

        Retourne :
        np.ndarray : Les prédictions. (sample, time, output_dim)
        """
        # Check if model is trained
        if self.model is None:
            raise Exception("Model is not trained")
        
        # Make numpy arrays
        X = np.array(X)

        # Run model
        preds = []
        for i in range(X.shape[0]):
            preds += [self.model.run(X[i])]
        preds = np.array(preds)

        return preds
        
    def count_params(self):
        """
        Compte le nombre de paramètres du modèle (appris et non-appris)

        Retourne :
        int : Le nombre de paramètres du modèle.
        """
        params = 0
        if self.reservoir.W is None or self.model is None:
            return params
        
        params += self.reservoir.Win.size
        params += self.reservoir.W.size
        params += self.reservoir.bias.size
        params += self.readout.Wout.size
        params += self.readout.bias.size
        return params
