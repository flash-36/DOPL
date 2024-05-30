import numpy as np


def compute_probabilities(R, comparisons):
    """Compute the probability matrix from R for all comparisons using vectorization."""
    exp_R = np.exp(R)
    idx_win = (comparisons[:, 0], comparisons[:, 1])
    idx_lose = (comparisons[:, 2], comparisons[:, 3])

    win_probs = exp_R[idx_win]
    lose_probs = exp_R[idx_lose]
    probabilities = win_probs / (win_probs + lose_probs)
    return probabilities


def neg_log_likelihood(R, comparisons):
    """Calculate the negative log-likelihood for the Bradley-Terry model using vectorization."""
    probabilities = compute_probabilities(R, comparisons)
    return -np.sum(
        np.log(probabilities + 1e-10)
    )  # Adding a small constant to prevent log(0)


def gradient(R, comparisons):
    """Vectorized gradient computation of the log-likelihood."""
    exp_R = np.exp(R)
    idx_win = (comparisons[:, 0], comparisons[:, 1])
    idx_lose = (comparisons[:, 2], comparisons[:, 3])

    win_probs = exp_R[idx_win]
    lose_probs = exp_R[idx_lose]
    probs = win_probs / (win_probs + lose_probs)

    grad = np.zeros_like(R)
    np.add.at(grad, idx_win, 1 - probs)
    np.subtract.at(grad, idx_lose, 1 - probs)

    return -grad  # Minimize negative log-likelihood


def mle_gradient_descent(comparisons, R_est, learning_rate=0.08, max_iter=6000):
    print("Calculating MLE...")
    R = R_est  # Estimate from previous iteration
    for iteration in range(max_iter):
        grad = gradient(R, comparisons)
        R -= (
            learning_rate * grad
        )  # Update R by stepping in the direction of the negative gradient
        if np.linalg.norm(grad) < 1e-6:  # Convergence criterion
            print(f"Converged at iteration {iteration}")
            break
    return R


if __name__ == "__main__":
    # Example data: (winning_arm, winning_state, losing_arm, losing_state)
    comparisons = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 1]])

    num_arms = 2
    num_states = 2
    R_est = np.ones((num_arms, num_states)) * 0.5  # Initial guess for R
    optimized_R = mle_gradient_descent(comparisons, R_est)
    print("Optimized R parameters:")
    print(optimized_R)
