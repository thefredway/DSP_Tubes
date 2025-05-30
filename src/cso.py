import numpy as np
import random

def fitness_snr(signal):
    """
    Menghitung invers dari Signal-to-Noise Ratio (SNR) sebagai fitness.
    Semakin rendah hasil return, semakin baik (karena kita minimasi -SNR).
    """
    snr = np.mean(signal)**2 / (np.std(signal)**2 + 1e-8)
    return -snr

def bandpass_and_eval(signal, fs, apply_filter, param_set):
    """
    Menerapkan bandpass filter pada sinyal dan mengembalikan nilai fitness.
    Params:
      signal       : sinyal 1D
      fs           : frame rate
      apply_filter : fungsi filtering
      param_set    : tuple (lowcut, highcut, order)
    Return:
      fitness (float)
    """
    lowcut, highcut, order = param_set
    if lowcut >= highcut or order < 2 or order > 8:
        return 1e9  # penalti besar jika parameternya tidak valid
    if len(signal) < 3 * fs:
        return 1e9  # sinyal terlalu pendek untuk filtering stabil
    try:
        filtered = apply_filter(signal, lowcut, highcut, fs, order=int(order))
        fitness_value = fitness_snr(filtered)
        if not np.isfinite(fitness_value):
            return 1e9
        return fitness_value
    except Exception:
        return 1e9

def cat_swarm_optimize(
    objective_func,
    bounds,
    n_cats=10,
    max_iter=30,
    mixture_ratio=0.5,
    srd=0.2,
    smp=5
):
    """
    Implementasi dasar algoritma Cat Swarm Optimization (CSO).
    Params:
      objective_func : fungsi objektif yang akan diminimalkan
      bounds         : list of tuples [(min1, max1), ...] untuk setiap dimensi
      n_cats         : jumlah populasi kucing
      max_iter       : jumlah iterasi
      mixture_ratio  : rasio antara seeking dan tracking mode (0–1)
      srd            : Seeking Range of the Dimension
      smp            : Seeking Memory Pool (jumlah kandidat per kucing)
    Return:
      best_cat (np.ndarray) : parameter terbaik
      best_score (float)    : nilai fitness terbaik
    """
    dim = len(bounds)
    cats = [np.array([np.random.uniform(*b) for b in bounds]) for _ in range(n_cats)]
    velocities = [np.zeros(dim) for _ in range(n_cats)]
    fitness = [objective_func(c) for c in cats]
    best_cat = cats[np.argmin(fitness)]
    best_score = min(fitness)

    for it in range(max_iter):
        for i in range(n_cats):
            if random.random() < mixture_ratio:
                # SEEKING MODE
                pool = []
                for _ in range(smp):
                    candidate = cats[i].copy()
                    for d in range(dim):
                        if random.random() < 0.5:
                            candidate[d] += np.random.uniform(-srd, srd) * (bounds[d][1] - bounds[d][0])
                            candidate[d] = np.clip(candidate[d], bounds[d][0], bounds[d][1])
                    pool.append(candidate)
                pool_fitness = [objective_func(p) for p in pool]
                cats[i] = pool[np.argmin(pool_fitness)]
                fitness[i] = min(pool_fitness)
            else:
                # TRACKING MODE
                velocities[i] += np.random.rand(dim) * (best_cat - cats[i])
                velocities[i] = np.clip(velocities[i], -0.1, 0.1)
                cats[i] += velocities[i]
                for d in range(dim):
                    cats[i][d] = np.clip(cats[i][d], bounds[d][0], bounds[d][1])
                fitness[i] = objective_func(cats[i])

        if min(fitness) < best_score:
            best_score = min(fitness)
            best_cat = cats[np.argmin(fitness)]

    return best_cat, best_score
