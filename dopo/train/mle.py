import numpy as np
from scipy.optimize import minimize


def neg_log_likelihood(R_flat, comparisons, num_arms, num_states):
    """Calculate the negative log-likelihood for the Bradley-Terry model using vectorized operations."""
    R = R_flat.reshape((num_arms, num_states))
    exp_R = np.exp(R)

    # Extract the indices for winners and losers
    win_indices = (comparisons[:, 0], comparisons[:, 1])
    lose_indices = (comparisons[:, 2], comparisons[:, 3])

    # Calculate the probabilities in a vectorized way
    win_probs = exp_R[win_indices]
    lose_probs = exp_R[lose_indices]
    total_probs = win_probs + lose_probs
    probabilities = win_probs / total_probs

    # Compute log-likelihood
    log_likelihood = np.sum(
        np.log(probabilities + 1e-20)
    )  # Small constant to avoid log(0)
    return -log_likelihood


def mle_bradley_terry(comparisons, R_est):
    """Estimate parameters using scipy.optimize.minimize with vectorized likelihood computation."""
    num_arms, num_states = R_est.shape
    initial_guess = R_est.flatten()  # Flatten R for optimization
    bounds = [(0, 1) for _ in range(num_arms * num_states)]

    print("Calculating MLE...")
    result = minimize(
        neg_log_likelihood,
        initial_guess,
        args=(comparisons, num_arms, num_states),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "gtol": 1e-3},
    )

    if result.success:
        return result.x.reshape((num_arms, num_states))  # Reshape R back to matrix form
    else:
        raise RuntimeError("Optimization did not converge: " + result.message)


if __name__ == "__main__":
    # Example usage
    comparisons = np.array(
        [
            (0, 1, 1, 0),  # (winning_arm, winning_state, losing_arm, losing_state)
            (1, 0, 0, 1),
            (0, 1, 1, 1),
        ]
    )

    num_arms = 2
    num_states = 2
    R_est = np.ones((num_arms, num_states)) * 0.5  # Initial guess for R
    optimized_R = mle_bradley_terry(comparisons, R_est)
    print("Optimized R parameters:")
    print(optimized_R)
