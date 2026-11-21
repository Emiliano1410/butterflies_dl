import numpy as np

PAIR = ("MONARCH", "VICEROY")
SPECIALIST_PATH = "results/especialista/best_model.keras"


def refinar(probs, spec_probs, classes):
    idx = [classes.index(c) for c in PAIR]
    rows = np.where(np.isin(probs.argmax(axis=1), idx))[0]
    probs = probs.copy()
    mass = probs[rows][:, idx].sum(axis=1, keepdims=True)
    probs[np.ix_(rows, idx)] = mass * spec_probs[rows]
    return probs
