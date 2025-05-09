import numpy as np
from datasets import load_dataset, load_from_disk


# ------------ USEFULL FUNCTIONS ------------ #

def compute_score(Y, Y_hat, prediction_timesteps, classification):
    """
    Compute the accuracy of the model.

    Parameters:
    - Y (np.ndarray): Target array [B, T, O]
    - Y_hat (np.ndarray): Predicted array [B, T, O]
    - prediction_timesteps (list): List of prediction timesteps
    - classification (bool): Whether the task is a classification task -> accuracy or MSE

    Returns:
    - accuracy (float): Accuracy value
    """
    # Make sure Y_hat and Y are numpy arrays
    if not isinstance(Y_hat, np.ndarray) or not isinstance(Y, np.ndarray):
        Y = np.array(Y, dtype=np.float32)
        Y_hat = np.array(Y_hat, dtype=np.float32)

    # Select only the prediction timesteps
    preds = []
    truths = []
    for j in range(Y.shape[0]):
        preds.append(Y_hat[j, prediction_timesteps[j], :])
        truths.append(Y[j, prediction_timesteps[j], :])

    if classification:
        # Compute the accuracy
        preds = np.argmax(np.stack(preds, axis=0), axis=-1)  # [B, prediction_timesteps] int: class
        truths = np.argmax(np.stack(truths, axis=0), axis=-1)  # [B, prediction_timesteps] int: class
        score = np.sum(preds == truths) / (truths.shape[0] * len(prediction_timesteps[0]))
        score = 1 - score

    else:
        # Compute the MSE
        preds = np.stack(preds, axis=0).reshape(-1, Y.shape[-1])  # [B * prediction_timesteps, O] float: logits
        truths = np.stack(truths, axis=0).reshape(-1, Y.shape[-1])
        score = np.mean((preds - truths) ** 2)

    return score

def _generate_train_test_samples(n_samples, training_ratio, generate_one_sample):
    # Generate the samples
    input, target, timesteps = zip(*[generate_one_sample() for _ in range(n_samples)])
    input, target, timesteps = np.array(input), np.array(target), np.array(timesteps)
    
    # Split the data into training and testing set
    training_size = int(n_samples * training_ratio)
    X_train = input[:training_size, :, :]
    Y_train = target[:training_size, :, :]
    T_train = timesteps[:training_size, :] # timesteps to predict
    X_test = input[training_size:, :, :]
    Y_test = target[training_size:, :, :]
    T_test = timesteps[training_size:, :] # timesteps to predict

    return X_train, Y_train, T_train, X_test, Y_test, T_test



# ------------ TEST DE MEMOIRE SIMPLE ------------ #

def generate_discrete_postcasting(n_samples=1000, sequence_length=1000, delay=10, n_symbols=8, training_ratio=0.8):
    """
    [Unique sequence]
    Génère une tâche de copie : le modèle doit reproduire la séquence d'entrée 
    (one-hot) après un délai.

    Args:
    - n_samples (int): nombre d'échantillons
    - sequence_length (int): longueur de la séquence
    - delay (int): délai avant de reproduire la séquence
    - n_symbols (int): nombre de symboles possibles
    - training_ratio (float): proportion de la séquence utilisée pour l'entraînement

    Return:
    - X_train (samples, sequence, n_symbols)
    - Y_train (samples, zero + sequence[:-delay], n_symbols)
    - T_train (samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    - X_test (samples, sequence, n_symbols)
    - Y_test (samples, zero + sequence[:-delay], n_symbols)
    - T_test (samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    """
    def generate_one_sample():
        # Generate the sequence
        sequence = np.random.randint(0, n_symbols, size=sequence_length)
        input = np.eye(n_symbols)[sequence].reshape(sequence_length, n_symbols)
        target = np.concatenate([np.zeros((delay, n_symbols)), input[:-delay, :]], axis=0)
        timesteps = np.arange(delay, sequence_length)

        return input, target, timesteps

    return _generate_train_test_samples(n_samples, training_ratio, generate_one_sample)




def generate_continue_postcasting(n_samples=1000, sequence_length=1000, delay=10, training_ratio=0.8):
    """
    [Unique sequence]
    Génère une tâche de copie : le modèle doit reproduire la séquence d'entrée 
    (continue) après un délai.

    Args:
    - n_samples (int): nombre d'échantillons
    - sequence_length (int): longueur de la séquence
    - delay (int): délai avant de reproduire la séquence
    - training_ratio (float): proportion de la séquence utilisée pour l'entraînement

    Return:
    - X_train (samples, sequence, 1)
    - Y_train (samples, delay + sequence[:-delay], 1)
    - T_train (samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    - X_test (samples, sequence, 1)
    - Y_test (samples, delay + sequence[:-delay], 1)
    - T_test (samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    """
    def generate_one_sample():
        # Generate the sequence
        input = np.random.uniform(-0.8, 0.8, size=sequence_length).reshape(sequence_length, 1)
        target = np.concatenate([np.zeros((delay, 1)), input[:-delay, :]], axis=0)
        timesteps = np.arange(delay, sequence_length)

        return input, target, timesteps

    return _generate_train_test_samples(n_samples, training_ratio, generate_one_sample)

# ------------ TEST DE TRAITEMENT DU SIGNAL ------------ #

def generate_sin_forecasting(sequence_length=1000, forecast_length=1, training_ratio=0.8):
    """
    [Unique sequence]
    Génère un signal sinusoïdal modulé en fréquence.
    Le modèle doit prédire la fréquence du signal à l'instant suivant.

    Args:
    - sequence_length (int): longueur de la séquence
    - forecast_length (int): longueur de la prédiction
    - training_ratio (float): proportion de la séquence utilisée pour l'entraînement

    Return:
    - X_train (1, training_sequence, 1)
    - Y_train (1, training_sequence, 1)
    - T_train (1, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    - X_test (1, testing_sequence, 1)
    - Y_test (1, testing_sequence, 1)
    - T_test (1, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    """
    # Generate the signal
    length = sequence_length + forecast_length
    max_value = length / 100
    t = np.linspace(0, max_value, length)
    carrier_freq = 10
    modulator_freq = 0.5
    modulator = np.sin(2 * np.pi * modulator_freq * t)
    carrier = np.sin(2 * np.pi * carrier_freq * t + modulator)

    # Create the input & target
    input = carrier[:-forecast_length].reshape(1, -1, 1)
    target = carrier[forecast_length:].reshape(1, -1, 1)

    # Split the data into training and testing set
    training_size = int(sequence_length * training_ratio)
    X_train = input[:, :training_size, :]
    Y_train = target[:, :training_size, :]
    X_test = input
    Y_test = target

    # Prediction timestep
    T_train = np.arange(forecast_length, training_size).reshape(1, -1)
    T_test = np.arange(training_size, sequence_length).reshape(1, -1)

    return X_train, Y_train, T_train, X_test, Y_test, T_test 

def generate_chaotic_forecasting(sequence_length=1000, forecast_length=1, training_ratio=0.8):
    """
    [Unique sequence]
    Génère une série temporelle chaotique (système de Lorenz).
    Le modèle doit prédire l'état du système à l'instant suivant.

    Args:
    - sequence_length (int): longueur de la séquence
    - forecast_length (int): longueur de la prédiction
    - training_ratio (float): proportion de sample utilisée pour l'entraînement

    Return:
    - X_train (1, training_sequence, 3)
    - Y_train (1, training_sequence, 3)
    - T_train (1, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    - X_test (1, testing_sequence, 3)
    - Y_test (1, testing_sequence, 3)
    - T_test (1, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    """
    # Define the Lorenz system
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        dx = s * (y - x)
        dy = r * x - y - x * z
        dz = x * y - b * z
        return dx, dy, dz
    
    # Generate the Lorenz system
    dt = 0.01
    stepCnt = sequence_length + forecast_length
    
    xs = np.zeros(stepCnt)
    ys = np.zeros(stepCnt)
    zs = np.zeros(stepCnt)
    
    xs[0], ys[0], zs[0] = (0., 1., 1.05)
    
    for i in range(stepCnt-1):
        dx, dy, dz = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (dx * dt)
        ys[i + 1] = ys[i] + (dy * dt)
        zs[i + 1] = zs[i] + (dz * dt)
    
    # Normalize the data
    xs = (xs - np.mean(xs)) / (3*np.std(xs))
    ys = (ys - np.mean(ys)) / (3*np.std(ys))
    zs = (zs - np.mean(zs)) / (3*np.std(zs))

    # Create the input & target
    input = np.column_stack((xs[:-forecast_length], ys[:-forecast_length], zs[:-forecast_length])).reshape(1, -1, 3)
    target = np.column_stack((xs[forecast_length:], ys[forecast_length:], zs[forecast_length:])).reshape(1, -1, 3)

    # Split the data into training and testing set
    training_size = int(sequence_length * training_ratio)
    X_train = input[:, :training_size, :]
    Y_train = target[:, :training_size, :]
    X_test = input
    Y_test = target

    # Prediction timestep
    T_train = np.arange(forecast_length, training_size).reshape(1, -1)
    T_test = np.arange(training_size, sequence_length).reshape(1, -1)

    return X_train, Y_train, T_train, X_test, Y_test, T_test

# ------------ TEST DE DEPENDANCE À LONG TERME ------------ #

def generate_discrete_pattern_completion(n_samples=1000, sequence_length=1000, n_symbols=8, base_length=5, mask_ratio=0.2, training_ratio=0.8):
    """
    [Multi sequence]
    Le modèle doit identifier et compléter des motifs répétitifs.
    La sequence consiste à répéter un motif de longueur base_length et de dimension n_symbols + 1.
    Le premier symbole est un marqueur indiquant quand le modèle doit prédire le motif.
    Les autres symboles sont des éléments du motif.

    Args:
    - n_samples (int): nombre d'échantillons
    - sequence_length (int): longueur de la séquence
    - n_symbols (int): nombre de symboles possibles
    - base_length (int): longueur du motif
    - mask_ratio (float): proportion de masquer un symbole
    - training_ratio (float): proportion de sample utilisée pour l'entraînement

    Return:
    - X_train (training_samples, sequence, n_symbols + marker)
    - Y_train (training_samples, sequence, n_symbols)
    - T_train (training_samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    - X_test (testing_samples, sequence, n_symbols + marker)
    - Y_test (testing_samples, sequence, n_symbols)
    - T_test (testing_samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    """
    def generate_one_sample():
        # Generate a base pattern
        base_pattern = np.random.randint(0, n_symbols, size=base_length)
        sequence = np.tile(base_pattern, sequence_length // base_length + 1)[:sequence_length]

        # Mask some parts so that the model predicts them
        nb_masked = int(sequence_length * mask_ratio)
        mask = np.random.choice(sequence_length, nb_masked, replace=False)
        masked_sequence = sequence.copy()
        masked_sequence[mask] = n_symbols

        # One-hot encoding
        input = np.eye(n_symbols+1)[masked_sequence]
        target = np.eye(n_symbols)[sequence]
        timesteps = mask

        return input, target, timesteps

    # Generate the samples
    return _generate_train_test_samples(n_samples, training_ratio, generate_one_sample)

def generate_continue_pattern_completion(n_samples=1000, sequence_length=100, base_length=5, mask_ratio=0.2, training_ratio=0.8):
    """
    [Multi sequence]
    Le modèle doit identifier et compléter des motifs répétitifs.
    La sequence consiste à répéter un motif de longueur base_length et de dimension 1.
    Certaines valeurs de la séquence sont masquées par la valeur -1 et le modèle doit les prédire.

    Args:
    - n_samples (int): nombre d'échantillons
    - sequence_length (int): longueur de la séquence
    - base_length (int): longueur du motif
    - mask_ratio (float): proportion de symboles masqués
    - training_ratio (float): proportion de sample utilisée pour l'entraînement

    Return:
    - X_train (training_samples, sequence, 1)
    - Y_train (training_samples, sequence, 1)
    - T_train (training_samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    - X_test (testing_samples, sequence, 1)
    - Y_test (testing_samples, sequence, 1)
    - T_test (testing_samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    """
    def generate_one_sample():
        # Generate a base pattern
        base_pattern = np.random.uniform(0, 1, size=base_length)
        sequence = np.tile(base_pattern, sequence_length // base_length + 1)[:sequence_length]

        # Mask some parts so that the model predicts them
        nb_masked = int(sequence_length * mask_ratio)
        mask = np.random.choice(sequence_length, nb_masked, replace=False)
        masked_sequence = sequence.copy()
        masked_sequence[mask] = -1

        # One-hot encoding
        input = masked_sequence.reshape(-1, 1)
        target = sequence.reshape(-1, 1)
        timesteps = mask

        return input, target, timesteps

    # Generate the samples
    return _generate_train_test_samples(n_samples, training_ratio, generate_one_sample)

def generate_copy_task(n_samples=1000, sequence_length=100, delay=10, n_symbols=8, training_ratio=0.8):
    """
    [Multi sequence]
    Génère une tâche de copie : le modèle doit lire l'ensemble d'une séquence, 
    la mémoriser et la reproduire après un délai, lorsqu'un trigger l'averti.

    Args:
    - n_samples (int): nombre d'échantillons
    - sequence_length (int): longueur de la séquence
    - delay (int): délai avant de reproduire la séquence
    - n_symbols (int): nombre de symboles possibles
    - training_ratio (float): proportion de sample utilisée pour l'entraînement

    Return:
    - X_train (train_samples, sequence + delay + 1 (marker) + zero_sequence, n_symbols + 1 (trigger))
    - Y_train (train_samples, zero_sequence + delay + 1 (marker) + sequence, n_symbols)
    - T_train (train_samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    - X_test (test_samples, sequence + delay + 1 (marker) + zero_sequence, n_symbols + 1 (trigger))
    - Y_test (test_samples, zero_sequence + delay + 1 (marker) + sequence, n_symbols)
    - T_test (test_samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    """
    def generate_one_sample(delay):
        # Generate a random sequence
        sequence = np.random.randint(0, n_symbols, size=sequence_length)  # 8 symboles possibles
        sequence_onehot = np.eye(n_symbols)[sequence]

        # Create the input & target
        input_sequence = np.zeros((sequence_length + delay + 1 + sequence_length, n_symbols + 1))
        input_sequence[:sequence_length, :n_symbols] = sequence_onehot
        input_sequence[sequence_length + delay, n_symbols] = 1  # marker

        target_sequence = np.zeros((sequence_length + delay + 1 + sequence_length, n_symbols))
        target_sequence[sequence_length + delay + 1:, :] = sequence_onehot

        timesteps = np.arange(sequence_length + delay + 1, sequence_length + delay + 1 + sequence_length)

        return input_sequence, target_sequence, timesteps
    
    # Generate the samples
    generate = lambda: generate_one_sample(delay)
    return _generate_train_test_samples(n_samples, training_ratio, generate)

def generate_selective_copy_task(n_samples=1000, sequence_length=100, delay=2, n_markers=2, n_symbols=8, training_ratio=0.8):
    """
    [Multi sequence]
    Le modèle doit lire l'ensemble d'une séquence, mémoriser les éléments marqués,
    et reproduire uniquement les éléments marqués dans la séquence, une fois le signal trigger reçu.

    Args:
    - n_samples (int): nombre d'échantillons
    - sequence_length (int): longueur de la séquence
    - delay (int): délai avant de reproduire la séquence
    - n_markers (int): nombre d'éléments à mémoriser < sequence_length
    - n_symbols (int): nombre de symboles possibles
    - training_ratio (float): proportion de sample utilisée pour l'entraînement

    Return: 
    - X_train (train_samples, sequence + delay + 1 (trigger) + zero_markers, n_symbols + 2 (marker + trigger))
    - Y_train (train_samples, zero_sequence + delay + 1 (trigger) + markers, n_symbols)
    - T_train (train_samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    - X_test (test_samples, sequence + delay + 1 (trigger) + zero_markers, n_symbols + 2 (marker + trigger))
    - Y_test (test_samples, zero_sequence + delay + 1 (trigger) + markers, n_symbols)
    - T_test (test_samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    """
    def generate_one_sample():
        # generate random sequence
        sequence = np.random.randint(0, n_symbols, size=sequence_length)
        sequence_onehot = np.eye(n_symbols)[sequence]
        selected_indices = np.random.choice(sequence_length, n_markers, replace=False)
        selected_indices.sort()

        # Create the input
        input = np.zeros((sequence_length + delay + 1 + n_markers, n_symbols + 2))
        input[:sequence_length, :n_symbols] = sequence_onehot # sequence
        input[selected_indices, n_symbols] = 1 # markers
        input[sequence_length + delay, n_symbols + 1] = 1

        # Create the target
        target = np.zeros((sequence_length + delay + 1 + n_markers, n_symbols))
        target[-n_markers:, :] = sequence_onehot[selected_indices, :]

        # Create the timesteps
        timesteps = np.arange(sequence_length + delay + 1, sequence_length + delay + 1 + n_markers)

        return input, target, timesteps

    # Generate the samples
    return _generate_train_test_samples(n_samples, training_ratio, generate_one_sample)

# ------------ TEST DE MANIPULATION DE L'INFORMATION RETENUE ------------ #

def generate_adding_problem(n_samples=1000, sequence_length=100, max_number=9, training_ratio=0.8):
    """
    [Multi sequence]
    Le modèle doit lire une séquence de nombre aléatoire, 
    puis additionner les nombres aux positions marquées une fois avoir reçu le signal trigger.

    Args:
    - n_samples (int): nombre d'échantillons
    - sequence_length (int): longueur de la séquence
    - max_number (int): nombre maximal possible
    - training_ratio (float): proportion de sample utilisée pour l'entraînement

    Return:
    - X_train (train_samples, sequence + trigger + 1, max_number + marker + trigger)
    - Y_train (train_samples, sequence + trigger + 1, 2*max_number - 1)
    - T_train (train_samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    - X_test (test_samples, sequence + trigger + 1, max_number + marker + trigger)
    - Y_test (test_samples, sequence + trigger + 1, 2*max_number - 1)
    - T_test (test_samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    """
    def generate_one_sample():
        # Génère la sequence
        sequence = np.random.randint(0, max_number, sequence_length)
        selected_indices = np.random.choice(sequence_length, 2, replace=False)
        result = (sequence[selected_indices] + 1).sum()

        # Create input
        input = np.zeros((sequence_length+2, max_number+2))
        input[:sequence_length, :max_number] = np.eye(max_number)[sequence] # One-hot encoding
        input[selected_indices, max_number] = 1 # Markers
        input[sequence_length, max_number+1] = 1 # Trigger

        # Create target
        target = np.zeros((sequence_length+2, max_number*2-1))
        target[-1, result-2] = 1

        # Create timesteps
        timesteps = np.arange(sequence_length+1, sequence_length+2)

        return input, target, timesteps
    
    # Generate the samples
    return _generate_train_test_samples(n_samples, training_ratio, generate_one_sample)

def generate_sorting_problem(n_samples=1000, sequence_length=100, n_symbols=8, training_ratio=0.8):
    """
    [Multi sequence]
    Génère une séquence de symbols (one-hot) de manière aléatoire, chacun étant associé à une position (one-hot). 
    Le modèle doit trier la séquence en fonction des positions, une fois le signal trigger reçu.

    Args:
    - n_samples (int): nombre d'échantillons
    - sequence_length (int): longueur de la séquence
    - n_symbols (int): nombre de symboles possibles
    - training_ratio (float): proportion de sample utilisée pour l'entraînement

    Return:
    - X_train (train_samples, sequence + trigger + zero_seq, n_symbols + order (sequence_length) + trigger)
    - Y_train (train_samples, zero_seq + trigger + sequence, n_symbols)
    - T_train (train_samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    - X_test (test_samples, sequence + trigger + zero_seq, n_symbols + order (sequence_length) + trigger)
    - Y_test (test_samples, zero_seq + trigger + sequence, n_symbols)
    - T_test (test_samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
    """
    def generate_one_sample():
        # Create a sequence of symbols & a random order
        sequence = np.random.randint(0, n_symbols, sequence_length)
        order = np.random.permutation(sequence_length)

        # One-hot encode the sequence and order
        sequence_onehot = np.eye(n_symbols)[sequence]
        order_onehot = np.eye(sequence_length + 1)[order]
        sequence_order = np.concatenate([sequence_onehot, order_onehot], axis=1)

        # Create other input parts   
        marker = np.zeros((1, n_symbols + sequence_length + 1))
        marker[0, n_symbols+sequence_length] = 1
        zero_input_pad = np.zeros((sequence_length, n_symbols + sequence_length + 1))

        # Create the input & target
        input = np.concatenate([sequence_order, marker, zero_input_pad], axis=0)
        target = np.zeros((sequence_length+1+sequence_length, n_symbols))
        target[sequence_length + 1 + order] = sequence_onehot

        # Create the timesteps
        timesteps = np.arange(sequence_length + 1, sequence_length + 1 + sequence_length)

        return input, target, timesteps
    
    # Generate the samples
    return _generate_train_test_samples(n_samples, training_ratio, generate_one_sample)

# def generate_mnist_classification(n_samples=1000, training_ratio=0.8, path=None):
#     """
#     [Multi sequence]
#     Génère une tâche de classification d'images MNIST : le modèle doit lire une image colonne par colonne,
#     la mémoriser et la classifier après un trigger.

#     Args:
#     - n_samples (int): nombre d'échantillons
#     - training_ratio (float): proportion de sample utilisée pour l'entraînement
#     - path (str): chemin vers le dataset MNIST, si None, le dataset est téléchargé

#     Return:
#     - X_train (train_samples, 28 + trigger + 1, 28 + trigger)
#     - Y_train (train_samples, 28 + trigger + 1, 10)
#     - T_train (train_samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
#     - X_test (test_samples, 28 + trigger + 1, 28 + trigger)
#     - Y_test (test_samples, 28 + trigger + 1, 10)
#     - T_test (test_samples, nb_timesteps) : timesteps où le modèle doit réaliser ses prédictions
#     """
#     # Load MNIST data
#     dataset = load_from_disk(path) if path else load_dataset("mnist")
#     X = np.concatenate([np.array(dataset['train']['image']), np.array(dataset['test']['image'])]).transpose(0, 2, 1) # so we can read it column by column
#     Y = np.concatenate([np.array(dataset['train']['label']), np.array(dataset['test']['label'])])
    
#     # Normalize the data
#     X = X / 255

#     # Shuffle and select the samples
#     shuffle = np.random.permutation(X.shape[0])[:n_samples]
#     X = X[shuffle]
#     Y = Y[shuffle]

#     # Create inputs
#     inputs = np.zeros((X.shape[0], X.shape[1]+2, X.shape[2]+1))
#     inputs[:, -2, -1] = 1 # trigger
#     inputs[:, :-2, :-1] = X

#     # Create targets
#     targets = np.zeros((X.shape[0], X.shape[1]+2, 10))
#     targets[:, -1, :] = np.eye(10)[Y]

#     # Split the data into training and testing set
#     training_size = int(n_samples * training_ratio)
#     X_train = inputs[:training_size]
#     Y_train = targets[:training_size]
#     X_test = inputs[training_size:]
#     Y_test = targets[training_size:]

#     # Prediction start
#     T_train = np.array([np.arange(29, 30) for _ in range(training_size)])
#     T_test = np.array([np.arange(29, 30) for _ in range(n_samples - training_size)])

#     return X_train, Y_train, T_train, X_test, Y_test, T_test

def generate_bracket_matching(n_samples=1000, sequence_length=100, max_depth=5, training_ratio=0.8):
    """
    [Multi sequence]
    Génère une séquence de parenthèses que le modèle doit valider.
    Test la capacité à maintenir un contexte hiérarchique.

    Args:
    - n_samples (int): nombre d'échantillons
    - sequence_length (int): longueur de la séquence
    - max_depth (int): profondeur maximale des parenthèses
    - training_ratio (float): proportion de sample utilisée pour l'entraînement

    Return:
    - X_train (training_samples, sequence + trigger + 1, 3)
    - Y_train (training_samples, sequence + trigger + 1, 1)
    - X_test (testing_samples, sequence + trigger + 1, 3)
    - Y_test (testing_samples, sequence + trigger + 1, 1)
    - prediction_timestep (list(int)) : liste des timesteps sur lesquels le model doit réaliser ses predictions
    """
    def generate_valid_sequence(length, max_depth):
        sequence = []
        stack = []
        remaining = length
        
        while remaining > 0:
            if len(stack) == 0 or (remaining > len(stack) and len(stack) < max_depth and np.random.random() > 0.5):
                sequence.append('(')
                stack.append('(')
            else:
                sequence.append(')')
                stack.pop()
            remaining -= 1
        
        return sequence

    def check_validity(sequence):
        stack = []
        for bracket in sequence:
            if bracket == '(':
                stack.append(bracket)
            elif len(stack) == 0:
                return 0
            else:
                stack.pop()
        return int(len(stack) == 0)

    def mutate_sequence(sequence, proba=0.35):
        nb_mutated = int(len(sequence) * proba)
        index = np.random.choice(len(sequence), nb_mutated, replace=False)
        mutation = ['(' if np.random.random() > 0.5 else ')' for _ in range(nb_mutated)]
        for i, bracket in zip(index, mutation):
            sequence[i] = bracket
        return sequence

    def generate_one_sample():
        # Generate a sequence
        sequence = generate_valid_sequence(sequence_length, max_depth)
        sequence = sequence if np.random.random() < 0.5 else mutate_sequence(sequence)
        validity = check_validity(sequence)

        # One-hot encode the sequence
        sequence_onehot = np.zeros((sequence_length+2, 3))
        for i, bracket in enumerate(sequence):
            sequence_onehot[i, 0 if bracket == '(' else 1] = 1
        sequence_onehot[-2, 2] = 1 # marker

        # Create the input & target
        input = sequence_onehot
        target = np.zeros((sequence_length+2, 2))
        target[-1, int(validity)] = 1

        # Create the timesteps
        timesteps = np.arange(sequence_length+1, sequence_length+2)

        return input, target, timesteps
    
    # Generate the samples
    return _generate_train_test_samples(n_samples, training_ratio, generate_one_sample)






