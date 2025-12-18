import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(
        self,
        k,
        dim,
        covariance_type="full",
        tol=1e-4,
        max_iter=100,
        reg_covar=1e-6,
        init_mu=None,
        init_sigma=None,
        init_pi=None
    ):
        self.k = k
        self.dim = dim
        self.covariance_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self.reg_covar = reg_covar

        self.mu = init_mu if init_mu is not None else np.random.randn(k, dim)
        self.pi = init_pi if init_pi is not None else np.ones(k) / k

        if init_sigma is not None:
            self.sigma = init_sigma
        else:
            self._init_covariances()

        self.log_likelihoods = []

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------
    def _init_covariances(self):
        if self.covariance_type == "full":
            self.sigma = np.array([np.eye(self.dim) for _ in range(self.k)])
        elif self.covariance_type == "tied":
            self.sigma = np.eye(self.dim)
        elif self.covariance_type == "diag":
            self.sigma = np.ones((self.k, self.dim))
        elif self.covariance_type == "spherical":
            self.sigma = np.ones(self.k)
        else:
            raise ValueError("Invalid covariance_type")

    # --------------------------------------------------
    # E-step
    # --------------------------------------------------
    def _e_step(self, X):
        N = X.shape[0]
        log_resp = np.zeros((N, self.k))

        for k in range(self.k):
            log_resp[:, k] = np.log(self.pi[k] + 1e-12) + self._log_gaussian(X, k)

        log_norm = self._logsumexp(log_resp, axis=1)
        self.z = np.exp(log_resp - log_norm[:, None])

    # --------------------------------------------------
    # M-step
    # --------------------------------------------------
    def _m_step(self, X):
        Nk = self.z.sum(axis=0)

        self.pi = Nk / X.shape[0]
        self.mu = (self.z.T @ X) / Nk[:, None]

        if self.covariance_type == "full":
            self.sigma = np.zeros((self.k, self.dim, self.dim))
            for k in range(self.k):
                diff = X - self.mu[k]
                self.sigma[k] = (
                    self.z[:, k][:, None] * diff
                ).T @ diff / Nk[k]
                self.sigma[k] += self.reg_covar * np.eye(self.dim)

        elif self.covariance_type == "tied":
            cov = np.zeros((self.dim, self.dim))
            for k in range(self.k):
                diff = X - self.mu[k]
                cov += (self.z[:, k][:, None] * diff).T @ diff
            self.sigma = cov / X.shape[0]
            self.sigma += self.reg_covar * np.eye(self.dim)

        elif self.covariance_type == "diag":
            self.sigma = np.zeros((self.k, self.dim))
            for k in range(self.k):
                diff = X - self.mu[k]
                self.sigma[k] = (
                    self.z[:, k][:, None] * diff**2
                ).sum(axis=0) / Nk[k] + self.reg_covar

        elif self.covariance_type == "spherical":
            self.sigma = np.zeros(self.k)
            for k in range(self.k):
                diff = X - self.mu[k]
                self.sigma[k] = (
                    self.z[:, k] * np.sum(diff**2, axis=1)
                ).sum() / (Nk[k] * self.dim) + self.reg_covar

    # --------------------------------------------------
    # Log Gaussian density
    # --------------------------------------------------
    def _log_gaussian(self, X, k):
        if self.covariance_type == "full":
            return multivariate_normal.logpdf(X, self.mu[k], self.sigma[k])

        elif self.covariance_type == "tied":
            return multivariate_normal.logpdf(X, self.mu[k], self.sigma)

        elif self.covariance_type == "diag":
            var = self.sigma[k]
            return -0.5 * (
                np.sum(np.log(2 * np.pi * var))
                + np.sum((X - self.mu[k])**2 / var, axis=1)
            )

        elif self.covariance_type == "spherical":
            var = self.sigma[k]
            return -0.5 * (
                self.dim * np.log(2 * np.pi * var)
                + np.sum((X - self.mu[k])**2, axis=1) / var
            )

    # --------------------------------------------------
    # Log-likelihood
    # --------------------------------------------------
    def log_likelihood(self, X):
        log_prob = np.zeros((X.shape[0], self.k))
        for k in range(self.k):
            log_prob[:, k] = np.log(self.pi[k] + 1e-12) + self._log_gaussian(X, k)
        return np.sum(self._logsumexp(log_prob, axis=1))

    # --------------------------------------------------
    # Fit EM
    # --------------------------------------------------
    def fit(self, X):
        prev_ll = None

        for _ in range(self.max_iter):
            self._e_step(X)
            self._m_step(X)

            ll = self.log_likelihood(X)
            self.log_likelihoods.append(ll)

            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    # --------------------------------------------------
    # Utility: log-sum-exp
    # --------------------------------------------------
    @staticmethod
    def _logsumexp(a, axis=None):
        a_max = np.max(a, axis=axis, keepdims=True)
        return np.squeeze(a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)))
