from collections.abc import Iterable

import numpy as np
from matplotlib import pyplot as plt
import matplotlib_tuda

matplotlib_tuda.load()
np.random.seed(12345)


def sample_prior(domain, K):
    samples = np.random.multivariate_normal(*prior(domain, K))

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
    fig.savefig(__file__.replace(".py", "") + "-" + filename + ".pdf")


def kernel_neural_network(p, q, Sigma):
    if len(p.shape) == 1:
        p = p.reshape((-1, 1))
    if len(q.shape) == 1:
        q = q.reshape((-1, 1))
    p = p[:, np.newaxis, :].repeat(repeats=q.shape[0], axis=1)
    q = q[np.newaxis, :, :].repeat(repeats=p.shape[0], axis=0)
    compute_tmp = lambda a, b: 2 * (np.einsum("nmi,ij,nmj->nm", a, Sigma[1:, 1:], b) + Sigma[0, 0])
    return (2 / np.pi) * np.arcsin(compute_tmp(p, q) / np.sqrt((1 + compute_tmp(p, p)) * (1 + compute_tmp(q, q))))


def prior(X, K):
    mean = np.zeros(len(X))
    cov = K(X, X)
    return mean, cov


def posterior(X, X_ast, f, K):
    training_weights = K(X_ast, X) @ np.linalg.inv(K(X, X))
    mean = training_weights @ f
    cov = K(X_ast, X_ast) - training_weights @ K(X, X_ast)
    return mean, cov


def main():
    kernels_1d = [
        ("s11", r"$\Sigma = \mathrm{diag}(1, 1)$", lambda p, q: kernel_neural_network(p, q, Sigma=np.diag([1, 1]))),
        ("s01", r"$\Sigma = \mathrm{diag}(0, 1)$", lambda p, q: kernel_neural_network(p, q, Sigma=np.diag([0, 1])))
    ]

    for name, sigma_name, K in kernels_1d:
        domain = np.arange(-5, 5, 0.1)
        actual_samples, prior_fn = sample_prior(domain, K)
        plot_distribution(domain, *prior(domain, K), actual_samples=actual_samples, title=f"Prior (NN Kernel, {sigma_name})", filename=f"prior-{name}")
        for training_points in [(0,), (-3, 0, 3)]:
            X, f = prior_fn(training_points)
            plot_distribution(domain, *posterior(X, domain, f, K), actual_samples=actual_samples, training_points=(X, f),
                              title=f"Posterior (NN Kernel, {sigma_name}, {len(training_points)} Data Points)", filename=f"posterior-{name}")


if __name__ == '__main__':
    main()
