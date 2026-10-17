from typing import TYPE_CHECKING, Self

from sklearn.base import BaseEstimator, MetaEstimatorMixin, TransformerMixin

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from sklearn.preprocessing import RobustScaler, StandardScaler


class ChannelScaler(BaseEstimator, TransformerMixin, MetaEstimatorMixin):
    def __init__(
        self,
        base_scaler: StandardScaler | RobustScaler,
    ) -> None:
        self.shape_: tuple[int]
        self.base_scaler = base_scaler

    def fit(
        self,
        X: npt.NDArray[np.floating],  # ruff: ignore[invalid-argument-name]
        y: None = None,  # ruff: ignore[unused-method-argument]
    ) -> Self:
        self.shape_ = X.shape
        self.base_scaler.fit(self._reshape_to_sklearn(X))
        return self

    def transform(
        self,
        X: npt.NDArray[np.floating],  # ruff: ignore[invalid-argument-name]
        y: None = None,  # ruff: ignore[unused-method-argument]
    ) -> npt.NDArray[np.floating]:
        x_transformed = self.base_scaler.transform(self._reshape_to_sklearn(X))
        return self._restore_from_sklearn(x_transformed)

    def fit_transform(
        self,
        X: npt.NDArray[np.floating],  # ruff: ignore[invalid-argument-name]
        y: None = None,  # ruff: ignore[unused-method-argument]
        **fit_kws,  # ruff: ignore[missing-type-kwargs, unused-method-argument]
    ) -> npt.NDArray[np.floating]:
        self.shape_ = X.shape
        x_reshaped = self._reshape_to_sklearn(X)
        self.base_scaler.fit(x_reshaped)
        return self._restore_from_sklearn(self.base_scaler.transform(x_reshaped))

    @staticmethod
    def _reshape_to_sklearn(
        x: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        return (
            x
            # (epochs, ..., times) -> (epochs, times, ...)
            .transpose(0, -1, *range(1, x.ndim - 1))
            # (epochs, times, ...) -> (epochs * times, -1)
            .reshape(x.shape[0] * x.shape[-1], -1)
        )

    def _restore_from_sklearn(
        self,
        x: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        n_epochs = int(x.shape[0] / self.shape_[-1])
        return (
            x
            # (epochs * times, -1) -> (epochs, times, ...)
            .reshape(n_epochs, self.shape_[-1], *self.shape_[1:-1])
            # (epochs, times, ...) -> (epochs, ..., times)
            .transpose(0, *range(2, len(self.shape_)), 1)
        )


class Vectorizer(BaseEstimator, TransformerMixin):
    """
    Transform (n_samples, n_features, ...) array into 3D array of (n_samples, n_features, n_parallelization).

    This class reshapes an n-dimensional array into an (n_samples, n_features,
    n_parallelization) array, usable by the estimators and transformers of
    scikit-learn, in conjunction with mne.decoding.SlidingEstimator and
    mne.decoding.GeneralizingEstimator.
    """

    def fit(
        self,
        X: npt.NDArray[np.floating],  # ruff: ignore[invalid-argument-name]
        y: None = None,  # ruff: ignore[unused-method-argument]
    ) -> Self:
        self.features_shape_ = X.shape[2:]
        return self

    def transform(self, X: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:  # ruff: ignore[invalid-argument-name]
        """Convert given array into two dimensions.

        Parameters
        ----------
        X : array-like
            The data to fit. Can be, for example a list, or an array of at least
            3d. The first dimension must be of length n_samples, where samples
            are the independent samples used by the estimator (e.g. n_epochs for
            epoched data). The last dimension must be of length n_times.

        Returns
        -------
        X : array, shape (n_samples, n_features, n_times)
            The transformed data.
        """
        return X.reshape(*X.shape[:2], -1)

    def fit_transform(
        self,
        X: npt.NDArray[np.floating],  # ruff: ignore[invalid-argument-name]
        y: None = None,  # ruff: ignore[unused-method-argument]
        **fit_kws,  # ruff: ignore[missing-type-kwargs, unused-method-argument]
    ) -> npt.NDArray[np.floating]:
        """Fit the data, then transform in one step.

        Parameters
        ----------
        X : array-like
            The data to fit. Can be, for example a list, or an array of at least
            3d. The first dimension must be of length n_samples, where samples
            are the independent samples used by the estimator (e.g. n_epochs for
            epoched data). The last dimension must be of length n_times.
        y : None | array, shape (n_samples,)
            Used for scikit-learn compatibility.

        Returns
        -------
        X : array, shape (n_samples, n_features, n_times)
            The transformed data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(
        self,
        X: npt.NDArray[np.floating],  # ruff: ignore[invalid-argument-name]
    ) -> npt.NDArray[np.floating]:
        """Transform 3D data back to its original feature shape.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features, n_times)
            Data to be transformed back to original shape.

        Returns
        -------
        X : array
            The data transformed into shape as used in fit. The first dimension
            is of length n_samples. The last dimension is of length n_times.
        """
        return X.reshape(*X.shape[:2], *self.features_shape_)
