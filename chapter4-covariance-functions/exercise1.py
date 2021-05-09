import os
from collections.abc import Iterable

import numpy as np
from matplotlib import pyplot as plt
import matplotlib_tuda

matplotlib_tuda.load()
np.random.seed(12345)


def sample_prior(domain):
    samples = np.random.multivariate_normal(*prior(domain))

    def prior_fn(xs):
        indices = []
        for x in (xs if isinstance(xs, Iterable) else [xs]):
            indices.append(int(np.where(np.isclose(x, domain))[0]))
        return domain[indices], samples[indices]

    return samples, prior_fn


def plot_distribution(domain, mean, cov, actual_samples=None, training_points=None, num_samples=3, title=None, filename=None):
    all_samples = np.random.multivariate_normal(mean, cov, size=num_samples)
    fig, ax = plt.subplots()
    ax.plot(domain, mean, label="Mean", zorder=1)
    ax.fill_between(domain, mean - 2 * np.diag(cov), mean + 2 * np.diag(cov), alpha=0.2, label="2x Std Dev", zorder=0)
    for samples in all_samples:
        ax.plot(domain, samples, ls="--", zorder=2)
    ax.plot([], [], ls="--", color="k", label="GP Samples")
    if actual_samples is not None:
        ax.scatter(domain, actual_samples, color="k", s=1, label="Actual Samples", zorder=3)
    if training_points is not None:
        ax.scatter(*training_points, color="k", s=100, marker="+", label="Training Points", zorder=4)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    ax.margins(x=0)
    fig.show()
    if filename is not None:
        fig.savefig("figures/" + os.path.basename(__file__).replace(".py", "") + "-" + filename + ".pdf")


def K(p, q):
    return np.exp(-np.abs(p[:, np.newaxis] - q) / 2)


def prior(X):
    mean = np.zeros(len(X))
    cov = K(X, X)
    return mean, cov


def posterior(X, X_ast, f):
    training_weights = K(X_ast, X) @ np.linalg.inv(K(X, X))
    mean = training_weights @ f
    cov = K(X_ast, X_ast) - training_weights @ K(X, X_ast)
    # Show training weights (OH process should be tri-diagonal).
    fig, ax = plt.subplots()
    mappable = ax.imshow(training_weights.T, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Test Data Points")
    ax.set_ylabel("Training Data Points")
    ax.set_title("Training Weights (Ornstein-Uhlenbeck)")
    fig.colorbar(mappable)
    fig.show()
    fig.savefig(__file__.replace(".py", "") + "-posterior-weights.pdf")
    return mean, cov


def main():
    domain = np.arange(-5, 5, 0.1)
    plot_distribution(domain, *prior(domain), title="Prior (Ornstein-Uhlenbeck)", filename="prior")
    X = np.array([-4, -2, 0, 2, 4])
    f = np.array([0.5, -0.5, 0, -0.5, 0.5])
    plot_distribution(domain, *posterior(X, domain, f), training_points=(X, f), title=f"Posterior (Ornstein-Uhlenbeck)", filename="posterior")


if __name__ == '__main__':
    main()
