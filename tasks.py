import numpy as np


# ------------ USEFULL FUNCTIONS ------------ #

def _generate_train_test_samples(n_samples, training_ratio, generate_one_sample):
    # Generate the samples
    input, target = zip(*[generate_one_sample() for _ in range(n_samples)])
    input, target = np.array(input), np.array(target)
    
    # Split the data into training and testing set
    training_size = int(n_samples * training_ratio)
    X_train = input[:training_size, :, :]
    Y_train = target[:training_size, :, :]
    X_test = input[training_size:, :, :]
    Y_test = target[training_size:, :, :]

    return X_train, Y_train, X_test, Y_test



# ------------ TEST DE MEMOIRE SIMPLE ------------ #

def generate_discrete_postcasting(sequence_length=1000, delay=10, n_symbols=8, training_ratio=0.8):
    """
    [Unique sequence]
    Génère une tâche de copie : le modèle doit reproduire la séquence d'entrée 
    (one-hot) après un délai.

    Args:
    - sequence_length (int): longueur de la séquence
    - delay (int): délai avant de reproduire la séquence
    - n_symbols (int): nombre de symboles possibles
    - training_ratio (float): proportion de la séquence utilisée pour l'entraînement

    Return:
    - X_train (1, training_sequence, n_symbols)
    - Y_train (1, delay + training_sequence[:-delay], n_symbols)
    - X_test (1, testing_sequence, n_symbols)
    - Y_test (1, delay + testing_sequence[:-delay], n_symbols)
    """
    # Compute the size of the training and testing set
    training_size = int(sequence_length * training_ratio)
    testing_size = sequence_length - training_size

    # Generate the sequence
    train_sequence = np.random.randint(0, n_symbols, size=training_size)
    test_sequence = np.random.randint(0, n_symbols, size=testing_size)

    # One hot encoding
    train_onehot = np.eye(n_symbols)[train_sequence].reshape(1, training_size, n_symbols)
    test_onehot = np.eye(n_symbols)[test_sequence].reshape(1, testing_size, n_symbols)

    # Create training and testing set
    X_train = train_onehot
    Y_train = np.concatenate([np.zeros((1, delay, n_symbols)), train_onehot[:, :-delay, :]], axis=1)
    X_test = test_onehot
    Y_test = np.concatenate([np.zeros((1, delay, n_symbols)), test_onehot[:, :-delay, :]], axis=1)
    
    return X_train, Y_train, X_test, Y_test

def generate_continue_postcasting(sequence_length=1000, delay=10, training_ratio=0.8):
    """
    [Unique sequence]
    Génère une tâche de copie : le modèle doit reproduire la séquence d'entrée 
    (continuous) après un délai.

    Args:
    - sequence_length (int): longueur de la séquence
    - delay (int): délai avant de reproduire la séquence
    - training_ratio (float): proportion de la séquence utilisée pour l'entraînement

    Return:
    - X_train (1, sequence, 1)
    - Y_train (1, delay + sequence[:-delay], 1)
    - X_test (1, sequence, 1)
    - Y_test (1, delay + sequence[:-delay], 1)
    """
    # Compute the size of the training and testing set
    training_size = int(sequence_length * training_ratio)
    testing_size = sequence_length - training_size

    # Generate the sequence
    training_sequence = np.random.uniform(-0.8, 0.8, size=training_size)
    test_sequence = np.random.uniform(-0.8, 0.8, size=testing_size)
    delay_sequence = np.zeros(delay)

    # Concatenate the sequence and the delay
    X_train = training_sequence.reshape(1, training_size, 1)
    Y_train = np.concatenate([delay_sequence, training_sequence[:-delay]]).reshape(1, training_size, 1)
    X_test = test_sequence.reshape(1, testing_size, 1)
    Y_test = np.concatenate([delay_sequence, test_sequence[:-delay]]).reshape(1, testing_size, 1)
    
    return X_train, Y_train, X_test, Y_test

def generate_copy_task(n_samples=1000, sequence_length=100, delay=10, n_symbols=8, training_ratio=0.8):
    """
    [Multi sequence]
    Génère une tâche de copie : le modèle doit lire l'ensemble d'une séquence, 
    la mémoriser et la reproduire après un délai, lorsqu'un signal l'averti.

    Args:
    - n_samples (int): nombre d'échantillons
    - sequence_length (int): longueur de la séquence
    - delay (int): délai avant de reproduire la séquence
    - n_symbols (int): nombre de symboles possibles
    - training_ratio (float): proportion de sample utilisée pour l'entraînement

    Return:
    - X_train (train_samples, sequence + delay + 1 (marker) + zero_sequence, n_symbols + 1 (signal))
    - Y_train (train_samples, zero_sequence + delay + 1 (marker) + sequence, n_symbols)
    - X_test (test_samples, sequence + delay + 1 (marker) + zero_sequence, n_symbols + 1 (signal))
    - Y_test (test_samples, zero_sequence + delay + 1 (marker) + sequence, n_symbols)
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

        return input_sequence, target_sequence
    
    # Generate the samples
    generate = lambda: generate_one_sample(delay)
    X_train, Y_train, X_test, Y_test = _generate_train_test_samples(n_samples, training_ratio, generate)
    
    return X_train, Y_train, X_test, Y_test

def generate_selective_copy_task(n_samples=1000, sequence_length=100, delay=2, n_markers=2, n_symbols=8, training_ratio=0.8):
    """
    [Multi sequence]
    Le modèle doit lire l'ensemble d'une séquence, mémoriser les éléments marqués,
    et reproduire uniquement les éléments marqués dans la séquence, lorsqu'un signal l'averti.

    Args:
    - n_samples (int): nombre d'échantillons
    - sequence_length (int): longueur de la séquence
    - n_markers (int): nombre d'éléments à mémoriser < sequence_length
    - n_symbols (int): nombre de symboles possibles
    - training_ratio (float): proportion de sample utilisée pour l'entraînement

    Return: 
    - X_train (train_samples, sequence + delay + 1 (signal) + zero_markers, n_symbols + 2 (marker + signal))
    - Y_train (train_samples, zero_sequence + delay + 1 (signal) + markers, n_symbols)
    - X_test (test_samples, sequence + delay + 1 (signal) + zero_markers, n_symbols + 2 (marker + signal))
    - Y_test (test_samples, zero_sequence + delay + 1 (signal) + markers, n_symbols)
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

        return input, target

    # Generate the samples
    generate = lambda: generate_one_sample()
    X_train, Y_train, X_test, Y_test = _generate_train_test_samples(n_samples, training_ratio, generate)

    return X_train, Y_train, X_test, Y_test





# ------------ TEST DE MANIPULATION DE L'INFORMATION RETENUE ------------ #

def generate_adding_problem(n_samples=1000, sequence_length=100, max_number=9, training_ratio=0.8):
    """
    [Multi sequence]
    Le modèle doit lire une séquence de nombre aléatoire, puis additionner les nombres aux positions marquées.

    Args:
    - n_samples (int): nombre d'échantillons
    - sequence_length (int): longueur de la séquence
    - max_number (int): nombre maximal possible
    - training_ratio (float): proportion de sample utilisée pour l'entraînement

    Return:
    - X_train (train_samples, sequence + trigger + 1, max_number + marker + trigger)
    - Y_train (train_samples, sequence + trigger + 1, 2*max_number - 1)
    - X_test (test_samples, sequence + trigger + 1, max_number + marker + trigger)
    - Y_test (test_samples, sequence + trigger + 1, 2*max_number - 1)
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

        return input, target
    
    # Generate the samples
    X_train, Y_train, X_test, Y_test = _generate_train_test_samples(n_samples, training_ratio, generate_one_sample)

    return X_train, Y_train, X_test, Y_test

def generate_sorting_problem(n_samples=1000, sequence_length=100, n_symbols=8, training_ratio=0.8):
    """
    [Multi sequence]
    Génère une séquence de symbols désordonné associé à un ordre. 
    Le modèle doit réordonner la séquence en fonction de l'ordre.

    Args:
    - n_samples (int): nombre d'échantillons
    - sequence_length (int): longueur de la séquence
    - n_symbols (int): nombre de symboles possibles
    - training_ratio (float): proportion de sample utilisée pour l'entraînement

    Return:
    - X_train (train_samples, sequence + trigger + zero_seq, n_symbols + order (sequence_length) + trigger)
    - Y_train (train_samples, zero_seq + trigger + sequence, n_symbols)
    - X_test (test_samples, sequence + trigger + zero_seq, n_symbols + order (sequence_length) + trigger)
    - Y_test (test_samples, zero_seq + trigger + sequence, n_symbols)
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

        return input, target
    
    # Generate the samples
    X_train, Y_train, X_test, Y_test = _generate_train_test_samples(n_samples, training_ratio, generate_one_sample)

    return X_train, Y_train, X_test, Y_test





# ------------ TEST DE DEPENDANCE À LONG TERME ------------ #

def generate_discrete_pattern_completion(sequence_length=1000, n_symbols=8, base_length=5, mask_ratio=0.2, training_ratio=0.8):
    """
    [Unique sequence]
    Le modèle doit identifier et compléter des motifs répétitifs.
    La sequence consiste à répéter un motif de longueur base_length et de dimension n_symbols + 1.
    Le premier symbole est un marqueur indiquant quand le modèle doit prédire le motif.
    Les autres symboles sont des éléments du motif.

    Args:
    - n_symbols (int): nombre de symboles possibles
    - base_length (int): longueur du motif
    - n_repetitions (int): nombre de répétitions du motif
    - mask_ratio (float): proportion de masquer un symbole
    - training_ratio (float): proportion de sample utilisée pour l'entraînement

    Return:
    - X_train (1, training_sequence, n_symbols + marker)
    - Y_train (1, training_sequence, n_symbols)
    - X_test (1, testing_sequence, n_symbols + marker)
    - Y_test (1, testing_sequence, n_symbols)
    """
    # Génère un motif de base
    base_pattern = np.random.randint(0, n_symbols, size=base_length)
    n_repetitions = sequence_length // base_length + 1
    sequence = np.tile(base_pattern, n_repetitions)[:sequence_length]

    # Masquer certaines parties pour que le modèle les prédise
    nb_masked = int(sequence_length * mask_ratio)
    mask = np.random.choice(sequence_length, nb_masked, replace=False)
    masked_sequence = sequence.copy()
    masked_sequence[mask] = n_symbols # Marker for masked values

    # One-hot encoding
    input = np.eye(n_symbols+1)[masked_sequence]
    target = np.eye(n_symbols)[sequence]

    # Split the data into training and testing set
    training_size = int(sequence_length * training_ratio)
    X_train = input[:training_size, :]
    Y_train = target[:training_size, :]
    X_test = input[training_size:, :]
    Y_test = target[training_size:, :]

    return X_train, Y_train, X_test, Y_test

def generate_continuous_pattern_completion(sequence_length=1000, base_length=5, mask_ratio=0.2, training_ratio=0.8):
    """
    [Unique sequence]
    Le modèle doit identifier et compléter des motifs répétitifs.

    Args:
    - base_length (int): longueur du motif
    - n_repetitions (int): nombre de répétitions du motif
    - mask_ratio (float): proportion de symboles masqués
    - training_ratio (float): proportion de sample utilisée pour l'entraînement

    Return:
    - X_train (1, training_sequence, 1)
    - Y_train (1, training_sequence, 1)
    - X_test (1, testing_sequence, 1)
    - Y_test (1, testing_sequence, 1)
    """
    # Génère un motif de base
    base_pattern = np.random.uniform(0, 1, size=base_length)
    n_repetitions = sequence_length // base_length + 1
    sequence = np.tile(base_pattern, n_repetitions)[:sequence_length]

    # Masquer certaines parties pour que le modèle les prédise
    nb_masked = int(sequence_length * mask_ratio)
    mask = np.random.choice(sequence_length, nb_masked, replace=False)
    masked_sequence = sequence.copy()
    masked_sequence[mask] = -1

    # Split the data into training and testing set
    input = masked_sequence.reshape(-1, 1)
    target = sequence.reshape(-1, 1)

    # Split the data into training and testing set
    training_size = int(sequence_length * training_ratio)
    testing_size = sequence_length - training_size
    X_train = input[:training_size, :].reshape(1, training_size, 1)
    Y_train = target[:training_size, :].reshape(1, training_size, 1)
    X_test = input[training_size:, :].reshape(1, testing_size, 1)
    Y_test = target[training_size:, :].reshape(1, testing_size, 1)

    return X_train, Y_train, X_test, Y_test

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
        target = np.zeros((sequence_length+2, 1))
        target[-1, 0] = int(validity)

        return input, target
    
    # Generate the samples
    X_train, Y_train, X_test, Y_test = _generate_train_test_samples(n_samples, training_ratio, generate_one_sample)

    return X_train, Y_train, X_test, Y_test





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
    - X_test (1, testing_sequence, 1)
    - Y_test (1, testing_sequence, 1)
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
    X_test = input[:, training_size:, :]
    Y_test = target[:, training_size:, :]

    return X_train, Y_train, X_test, Y_test

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
    - X_test (1, testing_sequence, 3)
    - Y_test (1, testing_sequence, 3)
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

    # Create the input & target
    input = np.column_stack((xs[:-forecast_length], ys[:-forecast_length], zs[:-forecast_length])).reshape(1, -1, 3)
    target = np.column_stack((xs[forecast_length:], ys[forecast_length:], zs[forecast_length:])).reshape(1, -1, 3)

    # Split the data into training and testing set
    training_size = int(sequence_length * training_ratio)
    X_train = input[:, :training_size, :]
    Y_train = target[:, :training_size, :]
    X_test = input[:, training_size:, :]
    Y_test = target[:, training_size:, :]

    return X_train, Y_train, X_test, Y_test    
