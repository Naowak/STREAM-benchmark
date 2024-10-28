import numpy as np

def generate_discrete_postcasting(sequence_length, delay_length, n_symbols=8):
    """
    Génère une tâche de copie : le modèle doit reproduire la séquence d'entrée 
    (one-hot) après un délai.

    Return:
    - input (sequence_length + delay_length, n_symbols + 1)
    - target (delay_length + sequence_length, n_symbols + 1)
    """
    input_sequence = np.random.randint(1, n_symbols+1, size=sequence_length)
    delay = np.zeros(delay_length)
    
    input_data = np.concatenate([input_sequence, delay]).astype(int)
    target_data = np.concatenate([delay, input_sequence]).astype(int)
    input_onehot = np.eye(n_symbols+1)[input_data]
    target_onehot = np.eye(n_symbols+1)[target_data]
    
    return input_onehot, target_onehot

def generate_continue_postcasting(sequence_length, delay_length):
    """
    Génère une tâche de copie : le modèle doit reproduire la séquence d'entrée 
    (continuous) après un délai.

    Return:
    - input (sequence_length + delay_length, 1)
    - target (delay_length + sequence_length, 1)
    """
    input_sequence = np.random.uniform(-0.8, 0.8, size=sequence_length)
    delay = np.zeros(delay_length)
    
    input_data = np.concatenate([input_sequence, delay]).reshape(-1, 1)
    target_data = np.concatenate([delay, input_sequence]).reshape(-1, 1)
    
    return input_data, target_data

def generate_copy_task(sequence_length, delay_length, n_symbols=8):
    """
    Génère une tâche de copie : le modèle doit lire l'ensemble d'une séquence, 
    la mémoriser et la reproduire après un délai, lorsqu'un signal l'averti.

    Return:
    - input (sequence_length + delay_length + 1 (marker) + zero_sequence, n_symbols + 1 (marker))
    - target (zero_sequence + delay_length + 1 (marker) + sequence, n_symbols + 1 (marker))
    """
    input_sequence = np.random.randint(1, n_symbols+1, size=sequence_length)  # 8 symboles possibles
    marker = n_symbols + 1  # marqueur de début de reproduction
    delay = np.zeros(delay_length)
    zero_sequence = np.zeros(sequence_length)
    
    input_data = np.concatenate([input_sequence, delay, [marker], zero_sequence]).astype(int)
    target_data = np.concatenate([zero_sequence, delay, [0], input_sequence]).astype(int)
    input_onehot = np.eye(n_symbols+2)[input_data]
    target_onehot = np.eye(n_symbols+2)[target_data]

    return input_onehot, target_onehot

def generate_selective_copy(sequence_length, n_markers=2, n_symbols=8):
    """
    Le modèle doit lire l'ensemble d'une séquence, mémoriser les éléments marqués,
    et reproduire uniquement les éléments marqués dans la séquence, lorsqu'un signal l'averti.

    Return: 
    - input (sequence_length + 1 (endflag) + n_markers (zero), n_symbols + 2 (pin and endflag))
    - target (sequence_length (zero) + 1 (endflag) + n_markers, n_symbols)
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

def generate_adding_problem(sequence_length, max_number=9):
    """
    Version classique : deux séquences parallèles
    - Une séquence de nombres aléatoires
    - Une séquence de marqueurs (deux 1, le reste 0)
    Le modèle doit additionner les nombres aux positions marquées
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






def generate_sorting_task(sequence_length):
    """
    Test la capacité à ordonner des informations
    """
    sequence = np.random.uniform(0, 1, size=sequence_length)
    target = np.sort(sequence)
    return sequence, target

def generate_bracket_matching(max_depth=5, sequence_length=100):
    """
    Génère une séquence de parenthèses que le modèle doit valider
    Test la capacité à maintenir un contexte hiérarchique
    """
    def generate_valid_sequence(length, max_depth):
        sequence = []
        stack = []
        remaining = length
        
        while remaining > 0:
            if len(stack) == 0 or (remaining > len(stack) and len(stack) < max_depth and random.random() > 0.5):
                sequence.append('(')
                stack.append('(')
            else:
                sequence.append(')')
                stack.pop()
            remaining -= 1
        
        return sequence
    
    sequence = generate_valid_sequence(sequence_length, max_depth)
    return sequence, 1  # 1 indique une séquence valide

def generate_pattern_completion(base_length=5, n_repetitions=4):
    """
    Le modèle doit identifier et compléter des motifs répétitifs
    """
    base_pattern = np.random.randint(0, 8, size=base_length)
    sequence = np.tile(base_pattern, n_repetitions)
    
    # Masquer certaines parties pour que le modèle les prédise
    mask = np.random.random(sequence.shape) > 0.3
    masked_sequence = sequence.copy()
    masked_sequence[~mask] = -1
    
    return masked_sequence, sequence

def generate_frequency_modulation():
    """
    Génère un signal modulé en fréquence
    Test la capacité à traiter des signaux complexes
    """
    t = np.linspace(0, 10, 1000)
    carrier_freq = 10
    modulator_freq = 0.5
    
    modulator = np.sin(2 * np.pi * modulator_freq * t)
    carrier = np.sin(2 * np.pi * carrier_freq * t + modulator)
    
    return {'signal': carrier[:-1]}, carrier[1:]

def generate_chaotic_series(length=1000):
    """
    Génère une série temporelle chaotique (système de Lorenz)
    """
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        dx = s * (y - x)
        dy = r * x - y - x * z
        dz = x * y - b * z
        return dx, dy, dz
    
    dt = 0.01
    stepCnt = length
    
    xs = np.zeros(stepCnt)
    ys = np.zeros(stepCnt)
    zs = np.zeros(stepCnt)
    
    xs[0], ys[0], zs[0] = (0., 1., 1.05)
    
    for i in range(stepCnt-1):
        dx, dy, dz = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (dx * dt)
        ys[i + 1] = ys[i] + (dy * dt)
        zs[i + 1] = zs[i] + (dz * dt)
    
    return {'x': xs[:-1], 'y': ys[:-1], 'z': zs[:-1]}, np.column_stack((xs[1:], ys[1:], zs[1:]))
