"""Model zoo: classical statistical baselines vs interpretable ML.

Every model exposes fit(X, y[, returns]) and predict_proba(X[, returns])
returning P(event). The GARCH baseline is the 'serious' statistical
benchmark: it ignores the feature matrix entirely and derives the
exceedance probability from a fitted AR-GJR-GARCH-t conditional
distribution — exactly what a classical risk desk would use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BaseRateModel:
    """Naive benchmark: predicts the training-set event frequency."""

    name = "base_rate"
    uses_features = False

    def fit(self, X, y, returns=None):
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X, returns=None):
        return np.full(len(X), self.p_)


class GarchVarModel:
    """AR(1)-GJR-GARCH(1,1)-t exceedance-probability baseline.

    P(r_{t+1} < VaR_t) computed from the model's one-step conditional
    distribution, where VaR_t is the same rolling empirical quantile
    used to define the label. Refit once per fold (no look-ahead).
    """

    name = "garch_t"
    uses_features = False

    def __init__(self, target_col: str = "cocoa", q: float = 0.05):
        self.target_col = target_col
        self.q = q

    def fit(self, X, y, returns: pd.DataFrame | None = None):
        r = returns.loc[: X.index[-1], self.target_col]
        self.res_ = arch_model(
            r, mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="t"
        ).fit(disp="off")
        return self

    def predict_proba(self, X, returns: pd.DataFrame | None = None):
        r_full = returns[self.target_col]
        # apply fitted parameters to the full history (fixed-parameter filter)
        res = arch_model(
            r_full, mean="AR", lags=1, vol="GARCH", p=1, o=1, q=1, dist="t"
        ).fix(self.res_.params.values)
        # one-step-ahead conditional mean and vol for each date t
        params = self.res_.params
        mu, phi = params["Const"], params[params.index[1]]  # AR(1) coef
        cond_vol = pd.Series(res.conditional_volatility, index=r_full.index)
        omega, alpha, gamma, beta = (params["omega"], params["alpha[1]"],
                                     params["gamma[1]"], params["beta[1]"])
        nu = params["nu"]
        resid = pd.Series(res.resid, index=r_full.index)
        h_next = (omega + (alpha + gamma * (resid < 0)) * resid**2
                  + beta * cond_vol**2)
        mu_next = mu + phi * r_full
        var_t = r_full.rolling(250).quantile(self.q)  # same threshold as label
        scale = np.sqrt(nu / (nu - 2.0))
        z = (var_t - mu_next) / np.sqrt(h_next)
        p = stats.t.cdf(z * scale, df=nu)
        return pd.Series(p, index=r_full.index).reindex(X.index).to_numpy()


class VarExceedanceModel:
    """VAR(p) exceedance-probability model: the multivariate linear
    statistical competitor GARCH cannot be, since GARCH is univariate by
    construction and blind to cross-market information (e.g. gold
    volatility predicting cedi risk) that a VAR can directly capture.

    Deliberately restricted to the five core daily return series
    (cocoa, gold, brent, wti, cedi), not the full feature set including
    dxy/vix: VAR parameters scale as k^2*p, and early expanding-window
    folds can have only a few hundred observations, so keeping k small
    is what makes the model estimable at all fold sizes rather than a
    convenience. A larger VAR including dxy/vix is a natural extension,
    noted as such rather than attempted here.

    Lag order p selected by AIC (capped at 5) on the training window;
    the one-step-ahead forecast error covariance for a VAR at h=1 is
    exactly the innovation covariance matrix (no MA expansion needed),
    so the target variable's one-step-ahead marginal distribution is
    directly available from the fitted VAR without simulation.
    """

    name = "var"
    uses_features = False

    def __init__(self, target_col: str = "cocoa", q: float = 0.05,
                max_lag: int = 5,
                core_cols=("cocoa", "gold", "brent", "wti", "cedi")):
        self.target_col = target_col
        self.q = q
        self.max_lag = max_lag
        self.core_cols = list(core_cols)

    def fit(self, X, y, returns: pd.DataFrame | None = None):
        from statsmodels.tsa.api import VAR

        r = returns.loc[: X.index[-1], self.core_cols]
        best_aic, best_res, best_p = np.inf, None, 1
        for p in range(1, self.max_lag + 1):
            try:
                res = VAR(r).fit(p)
            except Exception:
                continue
            if res.aic < best_aic:
                best_aic, best_res, best_p = res.aic, res, p
        self.res_ = best_res
        self.p_ = best_p
        self.target_idx_ = self.core_cols.index(self.target_col)
        return self

    def predict_proba(self, X, returns: pd.DataFrame | None = None):
        r_full = returns[self.core_cols]
        p = self.p_
        coefs = self.res_.coefs          # shape (p, k, k)
        intercept = self.res_.intercept  # shape (k,)
        sigma_u = self.res_.sigma_u_mlmiss if hasattr(self.res_, "sigma_u_mlmiss") \
            else self.res_.sigma_u
        sigma_target = np.sqrt(np.diag(sigma_u.values if hasattr(sigma_u, "values")
                                       else sigma_u)[self.target_idx_])

        vals = r_full.to_numpy()
        n = len(vals)
        mu_next = np.full(n, np.nan)
        for t in range(p, n):
            fc = intercept.copy()
            for lag in range(1, p + 1):
                fc = fc + coefs[lag - 1] @ vals[t - lag]
            mu_next[t] = fc[self.target_idx_]
        mu_next = pd.Series(mu_next, index=r_full.index)

        target_r = returns[self.target_col]
        var_t = target_r.rolling(250).quantile(self.q)  # same threshold as label
        z = (var_t - mu_next) / sigma_target
        prob = stats.norm.cdf(z)
        return pd.Series(prob, index=r_full.index).reindex(X.index).to_numpy()


class MarkovSwitchingVarModel:
    """Two-state Markov-switching exceedance-probability model.

    Directly reuses the validated regime-switching construction from the
    companion tail-dependence project's cedi robustness work: a 2-state
    Gaussian HMM fit to the target return series, with one-step-ahead
    PREDICTIVE regime probabilities (computed from the forward algorithm
    using information through t-1 only) weighting the mixture CDF used
    for the exceedance probability -- the same non-look-ahead
    construction validated there, applied here to forecasting rather
    than PIT diagnostics.
    """

    name = "markov_switching"
    uses_features = False

    def __init__(self, target_col: str = "cocoa", q: float = 0.05):
        self.target_col = target_col
        self.q = q

    def fit(self, X, y, returns: pd.DataFrame | None = None):
        from hmmlearn.hmm import GaussianHMM

        r = returns.loc[: X.index[-1], self.target_col].to_numpy().reshape(-1, 1)
        best_hmm, best_bic = None, np.inf
        for seed in range(5):
            hmm = GaussianHMM(n_components=2, covariance_type="diag",
                              n_iter=300, random_state=seed, tol=1e-6)
            try:
                hmm.fit(r)
            except Exception:
                continue
            ll = hmm.score(r)
            n_params = 2 + 2 * 2 + 2 * 2 - 1
            bic = -2 * ll + n_params * np.log(len(r))
            if bic < best_bic:
                best_bic, best_hmm = bic, hmm
        self.hmm_ = best_hmm
        means = self.hmm_.means_.ravel()
        self.quiet_ = int(np.argmin(np.sqrt(self.hmm_.covars_.ravel())))
        self.volatile_ = 1 - self.quiet_
        return self

    def predict_proba(self, X, returns: pd.DataFrame | None = None):
        hmm = self.hmm_
        means = hmm.means_.ravel()
        stds = np.sqrt(hmm.covars_.ravel())
        r_full = returns[self.target_col].to_numpy().reshape(-1, 1)
        n = len(r_full)

        log_emit = hmm._compute_log_likelihood(r_full)
        alpha = np.zeros((n, 2))
        pi_pred = np.zeros((n, 2))
        pi_pred[0] = hmm.get_stationary_distribution()
        a0 = pi_pred[0] * np.exp(log_emit[0])
        alpha[0] = a0 / a0.sum()
        for t in range(1, n):
            pi_pred[t] = alpha[t - 1] @ hmm.transmat_
            at = pi_pred[t] * np.exp(log_emit[t])
            s = at.sum()
            alpha[t] = at / s if s > 0 else pi_pred[t]

        target_r = returns[self.target_col]
        var_t = target_r.rolling(250).quantile(self.q).to_numpy()
        prob = np.empty(n)
        for t in range(n):
            prob[t] = (pi_pred[t, 0] * stats.norm.cdf(var_t[t], means[0], stds[0])
                      + pi_pred[t, 1] * stats.norm.cdf(var_t[t], means[1], stds[1]))
        return pd.Series(prob, index=returns.index).reindex(X.index).to_numpy()


def make_logistic():
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1e6)),  # ~unregularized
    ])


def make_lasso():
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1", solver="liblinear", C=0.1, max_iter=5000)),
    ])


def make_elastic_net():
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(
            solver="saga", l1_ratio=0.5, C=0.1, max_iter=5000)),
    ])


def make_random_forest():
    return RandomForestClassifier(
        n_estimators=400, min_samples_leaf=25, max_features="sqrt",
        n_jobs=-1, random_state=0)


def make_xgboost():
    from xgboost import XGBClassifier
    # min_child_weight is XGBoost's minimum-Hessian-weight-per-leaf
    # constraint, NOT the same quantity as sklearn's min_samples_leaf
    # (raw sample count) used elsewhere in this file -- for logistic
    # loss, Hessian weight per instance is p(1-p), so a value calibrated
    # against sample counts (e.g. 40) is far too restrictive here: with
    # event rates of 5-12% and training folds as small as ~1,150
    # observations, the smallest fold has as few as ~55 positive
    # examples total, and min_child_weight=40 prevented any tree from
    # ever splitting on minority-class signal, collapsing every
    # prediction to the training base rate. A much smaller value keeps
    # the constraint meaningful without eliminating the minority class.
    return XGBClassifier(
        max_depth=3, learning_rate=0.05, n_estimators=300,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", n_jobs=-1, random_state=0)


def make_svm():
    # probability=True enables predict_proba via internal Platt scaling
    # (a 5-fold CV inside the SVM's own .fit() call); scaling is essential
    # for SVM and is not optional the way it is for tree-based methods.
    from sklearn.svm import SVC
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale",
                    probability=True, random_state=0)),
    ])


def make_mlp():
    # Deliberately small and shallow: a single hidden layer with few
    # units, positioned explicitly as a weak benchmark per this
    # project's own scope ("neural networks as a weak benchmark, not
    # the centre"), not a competitor to the tree/boosting models.
    # early_stopping guards against overfitting given how few positive
    # examples exist in the smallest expanding-window folds (as few as
    # ~55), and a meaningful L2 penalty (alpha) does the same.
    from sklearn.neural_network import MLPClassifier
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(8,), activation="relu", alpha=1.0,
            solver="adam", max_iter=2000, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=15,
            random_state=0)),
    ])


def make_gbm():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300,
        min_samples_leaf=40, random_state=0)


class SklearnModel:
    uses_features = True

    def __init__(self, name: str, estimator):
        self.name = name
        self.est = estimator

    def fit(self, X, y, returns=None):
        self.est.fit(X, y)
        return self

    def predict_proba(self, X, returns=None):
        return self.est.predict_proba(X)[:, 1]


def model_zoo(target_col: str = "cocoa") -> list:
    return [
        BaseRateModel(),
        GarchVarModel(target_col=target_col),
        VarExceedanceModel(target_col=target_col),
        MarkovSwitchingVarModel(target_col=target_col),
        SklearnModel("logistic", make_logistic()),
        SklearnModel("lasso", make_lasso()),
        SklearnModel("elastic_net", make_elastic_net()),
        SklearnModel("random_forest", make_random_forest()),
        SklearnModel("xgboost", make_xgboost()),
        SklearnModel("svm", make_svm()),
        SklearnModel("mlp", make_mlp()),
    ]
