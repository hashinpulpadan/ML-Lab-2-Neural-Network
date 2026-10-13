"""
Lab 2: A Neural Network from Scratch
Dataset: sklearn load_digits
Architecture: 64 -> W -> W -> 10, with ReLU hidden layers and softmax output.
All forward and backward propagation calculations are implemented with NumPy.
"""

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATES = [0.001, 0.01, 0.1]
HIDDEN_WIDTHS = [16, 64]


def prepare_data():
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target.astype(np.int64)
    return train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )


def initialize_parameters(hidden_width, seed=42):
    rng = np.random.default_rng(seed)
    # Lab instructions specify weights initialized from N(0, 0.01).
    W1 = rng.normal(0, 0.01, size=(64, hidden_width))
    W2 = rng.normal(0, 0.01, size=(hidden_width, hidden_width))
    W3 = rng.normal(0, 0.01, size=(hidden_width, 10))
    b1 = np.zeros(hidden_width)
    b2 = np.zeros(hidden_width)
    b3 = np.zeros(10)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def forward(X, params):
    z1 = X @ params["W1"] + params["b1"]
    a1 = np.maximum(0.0, z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = np.maximum(0.0, z2)
    logits = a2 @ params["W3"] + params["b3"]
    probabilities = softmax(logits)
    cache = (X, z1, a1, z2, a2, probabilities)
    return probabilities, cache


def cross_entropy(probabilities, y):
    eps = 1e-12
    return float(-np.mean(np.log(probabilities[np.arange(len(y)), y] + eps)))


def backward(y, params, cache):
    X, z1, a1, z2, a2, probabilities = cache
    n = X.shape[0]
    dlogits = probabilities.copy()
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n

    dW3 = a2.T @ dlogits
    db3 = np.sum(dlogits, axis=0)

    da2 = dlogits @ params["W3"].T
    dz2 = da2 * (z2 > 0)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)

    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def loss_for_params(X, y, params):
    probabilities, _ = forward(X, params)
    return cross_entropy(probabilities, y)


def gradient_check(X, y, params, checks=10, epsilon=1e-4):
    """Central-difference check on 10 randomly selected weight entries."""
    analytical = backward(y, params, forward(X, params)[1])
    rng = np.random.default_rng(123)
    weight_names = ["W1", "W2", "W3"]
    entries = []
    for _ in range(checks):
        name = weight_names[int(rng.integers(0, len(weight_names)))]
        idx = tuple(int(rng.integers(0, dim)) for dim in params[name].shape)
        original = params[name][idx]

        params[name][idx] = original + epsilon
        loss_plus = loss_for_params(X, y, params)
        params[name][idx] = original - epsilon
        loss_minus = loss_for_params(X, y, params)
        params[name][idx] = original

        numerical = (loss_plus - loss_minus) / (2 * epsilon)
        analytic = analytical[name][idx]
        relative_error = abs(numerical - analytic) / max(
            1e-8, abs(numerical) + abs(analytic)
        )
        entries.append((name, idx, analytic, numerical, relative_error))
    passed = sum(row[4] < 1e-6 for row in entries)
    with open(OUTPUT_DIR / "gradient_check.txt", "w", encoding="utf-8") as f:
        f.write("Numerical gradient check (central difference)\n")
        f.write(f"epsilon = {epsilon}\n")
        f.write("Target: at least 9 of 10 relative errors < 1e-6\n\n")
        for name, idx, analytic, numerical, error in entries:
            f.write(
                f"{name}{idx}: analytical={analytic:.10e}, "
                f"numerical={numerical:.10e}, relative_error={error:.3e}\n"
            )
        f.write(f"\nPASS COUNT: {passed}/10\n")
        f.write(f"OVERALL: {'PASS' if passed >= 9 else 'CHECK NEEDED'}\n")
    print(f"Gradient check: {passed}/10 entries have relative error < 1e-6")
    return passed


def predict(X, params):
    probabilities, _ = forward(X, params)
    return np.argmax(probabilities, axis=1)


def train_model(X_train, y_train, hidden_width, learning_rate, seed=42):
    params = initialize_parameters(hidden_width, seed=seed)
    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(X_train))
        for start in range(0, len(X_train), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            X_batch, y_batch = X_train[indices], y_train[indices]
            probabilities, cache = forward(X_batch, params)
            grads = backward(y_batch, params, cache)
            for name in params:
                params[name] -= learning_rate * grads[name]
        full_probabilities, _ = forward(X_train, params)
        losses.append(cross_entropy(full_probabilities, y_train))
    return params, losses


def measure_timing(X, y, params, repeats=100):
    # Warm up first so initial one-off overhead is less influential.
    probabilities, cache = forward(X, params)
    backward(y, params, cache)

    forward_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _, cache = forward(X, params)
        forward_times.append(time.perf_counter() - start)

    backward_times = []
    for _ in range(repeats):
        probabilities, cache = forward(X, params)
        start = time.perf_counter()
        backward(y, params, cache)
        backward_times.append(time.perf_counter() - start)

    return float(np.median(forward_times) * 1000), float(np.median(backward_times) * 1000)


def main():
    X_train, X_test, y_train, y_test = prepare_data()
    print("Dataset loaded and prepared.")
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Input shape: {X_train.shape} | Test shape: {X_test.shape}")
    print("Input pixels scaled by 16; stratified 70/30 split; random_state=42.")

    # Check gradients on a small fixed batch using the 16-neuron architecture.
    check_params = initialize_parameters(16, seed=7)
    gradient_check(X_train[:8], y_train[:8], check_params)

    all_curves = {}
    rows = []
    timing_written = False
    for width in HIDDEN_WIDTHS:
        for lr in LEARNING_RATES:
            print(f"\nTraining hidden width={width}, learning rate={lr} ...")
            params, losses = train_model(
                X_train, y_train, width, lr, seed=RANDOM_STATE
            )
            test_accuracy = float(np.mean(predict(X_test, params) == y_test))
            train_accuracy = float(np.mean(predict(X_train, params) == y_train))
            all_curves[f"W={width}, lr={lr}"] = losses
            print(
                f"Final train loss={losses[-1]:.4f} | "
                f"train accuracy={train_accuracy:.4f} | test accuracy={test_accuracy:.4f}"
            )
            rows.append({
                "hidden_width": width,
                "learning_rate": lr,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "final_train_loss": losses[-1],
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
            })

            if not timing_written:
                fwd_ms, bwd_ms = measure_timing(X_train[:BATCH_SIZE], y_train[:BATCH_SIZE], params)
                with open(OUTPUT_DIR / "timing_results.txt", "w", encoding="utf-8") as f:
                    f.write("Median timings over 100 runs (batch size 32)\n")
                    f.write(f"Forward pass: {fwd_ms:.6f} ms\n")
                    f.write(f"Backward pass: {bwd_ms:.6f} ms\n\n")
                    f.write(
                        "The backward pass usually takes longer because it computes "
                        "gradients for each trainable weight and bias using matrix "
                        "multiplications; the forward pass computes activations and "
                        "probabilities. Actual timings depend on hardware and BLAS.\n"
                    )
                print(f"Timing: forward={fwd_ms:.6f} ms, backward={bwd_ms:.6f} ms")
                timing_written = True

    with open(OUTPUT_DIR / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(11, 7))
    for label, losses in all_curves.items():
        plt.plot(range(1, EPOCHS + 1), losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Training cross-entropy loss")
    plt.title("Neural Network from Scratch: Training Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curves.png", dpi=160)
    plt.close()

    print("\nAll experiments completed.")
    print("Saved evidence in the results folder:")
    print("- results/gradient_check.txt")
    print("- results/loss_curves.png")
    print("- results/timing_results.txt")
    print("- results/results.csv")
    print("\nReview these actual outputs before including them in your lab report.")


if __name__ == "__main__":
    main()
