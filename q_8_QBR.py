"""
QBR - Quantum Basis Regression
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize as scipy_minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
NUM_LAYERS = 3
MAXITER = 200


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def build_qbr_circuit(x_features, theta, n_qubits, n_layers):
    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        qc.ry(x_features[i], i)

    idx = 0
    for layer in range(n_layers):
        for i in range(n_qubits):
            qc.ry(theta[idx], i)
            idx += 1
        for i in range(n_qubits):
            qc.rz(theta[idx], i)
            idx += 1
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        if layer % 2 == 1:
            qc.cx(n_qubits - 1, 0)

    return qc


def num_params():
    return NUM_LAYERS * NUM_QUBITS * 2


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def predict_single(x, theta):
    qc = build_qbr_circuit(x, theta, NUM_QUBITS, NUM_LAYERS)
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()
    n_states = 1 << NUM_QUBITS
    indices = np.arange(n_states, dtype=float)
    return float(np.dot(probs, indices)) / (n_states - 1)


def predict_all(X, theta):
    return np.array([predict_single(x, theta) for x in X])


def train_qbr(X, y):
    n_p = num_params()
    theta0 = np.random.uniform(0, 2 * np.pi, n_p)

    def loss(theta):
        preds = predict_all(X, theta)
        return float(np.mean((preds - y) ** 2))

    res = scipy_minimize(loss, theta0, method='COBYLA',
                         options={'maxiter': MAXITER, 'rhobeg': 0.5})
    return res.x, res.fun


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    n_states = 1 << NUM_QUBITS
    X_all = np.array([value_to_features(v) for v in range(n_states)])

    print(f"\n--- QBR ({NUM_QUBITS}q, {NUM_LAYERS} sloja, Ry+Rz+CX, "
          f"COBYLA {MAXITER} iter) ---")
    print(f"  Parametara po modelu: {num_params()}")

    dists = []
    for pos in range(7):
        print(f"  Poz {pos+1}...", end=" ", flush=True)
        y = build_empirical(draws, pos)

        theta, final_loss = train_qbr(X_all, y)

        pred = predict_all(X_all, theta)
        pred = pred - pred.min()
        if pred.sum() > 0:
            pred /= pred.sum()
        dists.append(pred)

        top_idx = np.argsort(pred)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{pred[i]:.3f}" for i in top_idx)
        print(f"loss={final_loss:.6f}  top: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QBR, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()



"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- QBR (5q, 3 sloja, Ry+Rz+CX, COBYLA 200 iter) ---
  Parametara po modelu: 30
  Poz 1... loss=0.111475  top: 17:0.055 | 18:0.052 | 8:0.048
  Poz 2... loss=0.102012  top: 2:0.092 | 3:0.076 | 4:0.066
  Poz 3... loss=0.105957  top: 34:0.075 | 17:0.061 | 16:0.053
  Poz 4... loss=0.079860  top: 4:0.102 | 35:0.091 | 34:0.080
  Poz 5... loss=0.119632  top: 9:0.069 | 8:0.067 | 21:0.061
  Poz 6... loss=0.108218  top: 37:0.058 | 27:0.052 | 28:0.052
  Poz 7... loss=0.117143  top: 38:0.077 | 22:0.056 | 21:0.055

==================================================
Predikcija (QBR, deterministicki, seed=39):
[17, 31, 34, 35, 36, 37, 38]
==================================================
"""



"""
QBR - Quantum Basis Regression

Rucno gradjeno kolo: Ry enkodiranje + 3 sloja (Ry + Rz rotacije + CX entanglement sa ciklicnim vezama)
Izlaz: ocekivana vrednost indeksa iz Born distribucije - sum(probs * indices) 
- kolo direktno predvidja "gde" u prostoru stanja je odgovor
Bogatija ekspresivnost: Ry + Rz daje punu rotaciju na Bloch sferi (za razliku od samo Ry u TwoLocal)
30 parametara (3 sloja x 5 qubita x 2 rotacije), COBYLA 200 iter
Egzaktno, deterministicki, Statevector

QBR = Quantum Basis Regression 
– koristi custom basis kolo, predikcija preko očekivane vrednosti indeksa u Born distribuciji
"""
