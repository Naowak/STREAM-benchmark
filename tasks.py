import numpy as np

# ------------ TEST DE MEMOIRE SIMPLE ------------ #

def generate_discrete_postcasting(sequence_length=1000, delay=10, n_symbols=8):
    """
    [Unique sequence]
    Génère une tâche de copie : le modèle doit reproduire la séquence d'entrée 
    (one-hot) après un délai.

    Args:
    - sequence_length (int): longueur de la séquence
    - delay (int): délai avant de reproduire la séquence
    - n_symbols (int): nombre de symboles possibles

    Return:
    - input (sequence + delay, n_symbols + 1)
    - target (delay + sequence, n_symbols + 1)
    """
    input_sequence = np.random.randint(1, n_symbols+1, size=sequence_length)
    delay = np.zeros(delay)
    
    input_data = np.concatenate([input_sequence, delay]).astype(int)
    target_data = np.concatenate([delay, input_sequence]).astype(int)
    input_onehot = np.eye(n_symbols+1)[input_data]
    target_onehot = np.eye(n_symbols+1)[target_data]
    
    return input_onehot, target_onehot

def generate_continue_postcasting(sequence_length=1000, delay=10):
    """
    [Unique sequence]
    Génère une tâche de copie : le modèle doit reproduire la séquence d'entrée 
    (continuous) après un délai.

    Args:
    - sequence_length (int): longueur de la séquence
    - delay (int): délai avant de reproduire la séquence

    Return:
    - input (sequence + delay, 1)
    - target (delay + sequence, 1)
    """
    input_sequence = np.random.uniform(-0.8, 0.8, size=sequence_length)
    delay = np.zeros(delay)
    
    input_data = np.concatenate([input_sequence, delay]).reshape(-1, 1)
    target_data = np.concatenate([delay, input_sequence]).reshape(-1, 1)
    
    return input_data, target_data

def generate_copy_task(sequence_length=1000, delay=10, n_symbols=8):
    """
    [Multi sequence]
    Génère une tâche de copie : le modèle doit lire l'ensemble d'une séquence, 
    la mémoriser et la reproduire après un délai, lorsqu'un signal l'averti.

    Args:
    - sequence_length (int): longueur de la séquence
    - delay (int): délai avant de reproduire la séquence
    - n_symbols (int): nombre de symboles possibles

    Return:
    - input (sequence + delay + 1 (marker) + zero_sequence, n_symbols + 1 (marker))
    - target (zero_sequence + delay + 1 (marker) + sequence, n_symbols + 1 (marker))
    """
    input_sequence = np.random.randint(1, n_symbols+1, size=sequence_length)  # 8 symboles possibles
    marker = n_symbols + 1  # marqueur de début de reproduction
    delay = np.zeros(delay)
    zero_sequence = np.zeros(sequence_length)
    
    input_data = np.concatenate([input_sequence, delay, [marker], zero_sequence]).astype(int)
    target_data = np.concatenate([zero_sequence, delay, [0], input_sequence]).astype(int)
    input_onehot = np.eye(n_symbols+2)[input_data]
    target_onehot = np.eye(n_symbols+2)[target_data]

    return input_onehot, target_onehot

def generate_selective_copy_task(sequence_length=1000, n_markers=2, n_symbols=8):
    """
    [Multi sequence]
    Le modèle doit lire l'ensemble d'une séquence, mémoriser les éléments marqués,
    et reproduire uniquement les éléments marqués dans la séquence, lorsqu'un signal l'averti.

    Args:
    - sequence_length (int): longueur de la séquence
    - n_markers (int): nombre d'éléments à mémoriser
    - n_symbols (int): nombre de symboles possibles

    Return: 
    - input (sequence + 1 (endflag) + n_markers (zero), n_symbols + 2 (pin and endflag))
    - target (zero_sequence + 1 (endflag) + n_markers, n_symbols)
    """
    # generate random onehot sequence
    sequence = np.random.randint(0, n_symbols, size=sequence_length)
    sequence_onehot = np.eye(n_symbols)[sequence]
    sequence_padded = np.concatenate([sequence_onehot, np.zeros((1+n_markers, n_symbols))], axis=0)
    sequence_padded

    # selection column
    selected_indices = np.random.choice(sequence_length, n_markers, replace=False)
    selection = np.zeros(sequence_length + n_markers + 1).reshape(-1, 1)
    selection[selected_indices, 0] = 1

    # end flag column
    endflag = np.zeros(sequence_length + n_markers + 1).reshape(-1, 1)
    endflag[sequence_length, 0] = 1
    input = np.concatenate([sequence_padded, selection, endflag], axis=1)

    # Compute target
    target = np.zeros((sequence_length + n_markers + 1, n_symbols))
    target[-n_markers:, :] = sequence_onehot[selected_indices, :]

    return input, target





# ------------ TEST DE MANIPULATION DE L'INFORMATION RETENUE ------------ #

def generate_adding_problem(sequence_length=1000, max_number=9):
    """
    [Multi sequence]
    Version classique : deux séquences parallèles
    - Une séquence de nombres aléatoires
    - Une séquence de marqueurs (deux 1, le reste 0)
    Le modèle doit additionner les nombres aux positions marquées

    Args:
    - sequence_length (int): longueur de la séquence
    - max_number (int): nombre maximal possible

    Return:
    - input (sequence + endflag + zero (1), max_number + marker + endflag)
    - target (zero_sequence + endflag + target, max_number * 2 + 1)
    """
    # Génère une séquence de nombres aléatoires
    sequence = np.random.randint(0, max_number+1, sequence_length)
    sequence_onehot = np.eye(max_number+1)[sequence]
    sequence_padded = np.concatenate([sequence_onehot, np.zeros((2, max_number+1))], axis=0)

    # Place deux marqueurs aléatoirement
    selected_indices = np.random.choice(sequence_length, 2, replace=False)
    selection = np.zeros(sequence_length + 2).reshape(-1, 1)
    selection[selected_indices] = 1

    # End flag column
    endflag = np.zeros(sequence_length + 2).reshape(-1, 1)
    endflag[sequence_length, 0] = 1
    input = np.concatenate([sequence_padded, selection, endflag], axis=1)

    # Calcule la somme des nombres aux positions marquées
    result = sequence[selected_indices].sum()
    target = np.zeros((sequence_length + 2, max_number * 2 + 1))
    target[-1, result] = 1

    return input, target

def generate_sorting_problem(sequence_length=1000, n_symbols=8):
    """
    [Multi sequence]
    Génère une séquence de symbols désordonné associé à un ordre. 
    Le modèle doit réordonner la séquence en fonction de l'ordre.

    Args:
    - sequence_length (int): longueur de la séquence
    - n_symbols (int): nombre de symboles possibles

    Return:
    - input (sequence + 1 + zero_seq, n_symbols + 1 + order (sequence_length))
    - target (zero_seq + 1 + sequence, n_symbols)
    """

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





# ------------ TEST DE DEPENDANCE À LONG TERME ------------ #

def generate_discrete_pattern_completion(n_symbols=8, base_length=5, n_repetitions=4, proba_mask=0.2):
    """
    [Unique sequence]
    Le modèle doit identifier et compléter des motifs répétitifs.

    Args:
    - n_symbols (int): nombre de symboles possibles
    - base_length (int): longueur du motif
    - n_repetitions (int): nombre de répétitions du motif
    - proba_mask (float): probabilité de masquer un symbole

    Return:
    - input (sequence, n_symbols + 1)
    - target (sequence, n_symbols)
    """
    base_pattern = np.random.randint(1, n_symbols+1, size=base_length)
    sequence = np.tile(base_pattern, n_repetitions)

    # Masquer certaines parties pour que le modèle les prédise
    mask = np.random.random(sequence.shape) < proba_mask
    masked_sequence = sequence.copy()
    masked_sequence[mask] = 0

    input = np.eye(n_symbols+1)[masked_sequence]
    target = np.eye(n_symbols+1)[sequence][:, 1:]

    return input, target

def generate_continuous_pattern_completion(base_length=5, n_repetitions=4, proba_mask=0.2):
    """
    [Unique sequence]
    Le modèle doit identifier et compléter des motifs répétitifs.

    Args:
    - base_length (int): longueur du motif
    - n_repetitions (int): nombre de répétitions du motif
    - proba_mask (float): probabilité de masquer un symbole

    Return:
    - input (sequence, 1)
    - target (sequence, 1)
    """
    base_pattern = np.random.uniform(0, 1, size=base_length)
    sequence = np.tile(base_pattern, n_repetitions)

    # Masquer certaines parties pour que le modèle les prédise
    mask = np.random.random(sequence.shape) < proba_mask
    masked_sequence = sequence.copy()
    masked_sequence[mask] = -1

    input = masked_sequence.reshape(-1, 1)
    target = sequence.reshape(-1, 1)

    return input, target

def generate_bracket_matching(sequence_length=100, max_depth=5):
    """
    [Multi sequence]
    Génère une séquence de parenthèses que le modèle doit valider.
    Test la capacité à maintenir un contexte hiérarchique.

    Args:
    - sequence_length (int): longueur de la séquence
    - max_depth (int): profondeur maximale des parenthèses

    Return:
    - input (sequence + 2, 3)
    - target (sequence + 2, 1)
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




# ------------ TEST DE TRAITEMENT DU SIGNAL ------------ #

def generate_sin_forecasting(sequence_length=1000, forecast_length=1):
    """
    [Unique sequence]
    Génère un signal sinusoïdal modulé en fréquence.
    Le modèle doit prédire la fréquence du signal à l'instant suivant.

    Args:
    - sequence_length (int): longueur de la séquence
    - forecast_length (int): longueur de la prédiction

    Return:
    - input (sequence, 1)
    - target (sequence, 1)
    """
    length = sequence_length + forecast_length
    max_value = length / 100
    t = np.linspace(0, max_value, length)
    carrier_freq = 10
    modulator_freq = 0.5

    modulator = np.sin(2 * np.pi * modulator_freq * t)
    carrier = np.sin(2 * np.pi * carrier_freq * t + modulator)

    input = carrier[:-forecast_length].reshape(-1, 1)
    target = carrier[forecast_length:].reshape(-1, 1)

    return input, target

def generate_chaotic_forecasting(sequence_length=1000, forecast_length=1):
    """
    [Unique sequence]
    Génère une série temporelle chaotique (système de Lorenz).
    Le modèle doit prédire l'état du système à l'instant suivant.

    Args:
    - sequence_length (int): longueur de la séquence
    - forecast_length (int): longueur de la prédiction

    Return:
    - input (sequence, 3)
    - target (sequence, 3)
    """
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        dx = s * (y - x)
        dy = r * x - y - x * z
        dz = x * y - b * z
        return dx, dy, dz
    
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

    input = np.column_stack((xs[:-forecast_length], ys[:-forecast_length], zs[:-forecast_length]))
    target = np.column_stack((xs[forecast_length:], ys[forecast_length:], zs[forecast_length:]))

    return input, target
    
