import os
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


def plot_length_scale(domain, length_scale_fn, name=None, filename=None):
    fig, ax = plt.subplots()
    length_scale = length_scale_fn(domain)
    ax.plot(domain, np.array((length_scale,)).repeat(len(domain)) if type(length_scale) == float else length_scale, label="Length Scale", zorder=1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$\ell(x)$")
    if name is not None:
        ax.set_title("Length Scale" + ("" if name is None else f" ({name})"))
    ax.legend()
    # ax.margins(x=0)
    fig.show()
    if filename is not None:
        fig.savefig("figures/" + os.path.basename(__file__).replace(".py", "") + "-" + filename + ".pdf")


def kernel_gibbs(p, q, length_scale_fn):
    if len(p.shape) == 1:
        p = p.reshape((-1, 1))
    if len(q.shape) == 1:
        q = q.reshape((-1, 1))
    p = p[:, np.newaxis, :].repeat(repeats=q.shape[0], axis=1)
    q = q[np.newaxis, :, :].repeat(repeats=p.shape[0], axis=0)
    length_scale_p = length_scale_fn(p)
    length_scale_q = length_scale_fn(q)
    normalization = np.sqrt(2 * length_scale_p * length_scale_q / (length_scale_p ** 2 + length_scale_q ** 2)).prod(axis=-1)
    sum = ((p - q) ** 2 / (length_scale_p ** 2 + length_scale_q ** 2)).sum(axis=-1)
    return normalization * np.exp(-sum)


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
    kernels = [
        ("const", r"$\ell(x) = \mathrm{const}$", lambda x: 1.0),
        ("squared", r"$\ell(x) = x^2$", lambda x: x ** 2),
        ("sin", r"$\ell(x) = \sin(x) + 1$", lambda x: np.sin(x / 2) + 1),
        ("sin-reciprocal", r"$\ell(x) = \sin(1/x) + 0.01$", lambda x: np.sin(1 / x) + 1.01),
        ("gaussian", r"$\ell(x) = \mathcal{N}(x \,\vert\, 0, 1)$", lambda x: np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)),
        ("inverted-gaussian", r"$\ell(x) = 0.5 - \mathcal{N}(x \,\vert\, 0, 1)$", lambda x: 0.5 - np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi))
    ]

    for name, length_scale_name, length_scale_fn in kernels:
        K = lambda p, q: kernel_gibbs(p, q, length_scale_fn)
        domain = np.arange(-5, 5, 0.1)
        plot_length_scale(domain, length_scale_fn, name=length_scale_name, filename=f"length-scale-{name}")
        actual_samples, prior_fn = sample_prior(domain, K)
        plot_distribution(domain, *prior(domain, K), actual_samples=actual_samples, title=f"Prior (Gibbs Kernel, {length_scale_name})", filename=f"prior-{name}")


if __name__ == '__main__':
    main()
