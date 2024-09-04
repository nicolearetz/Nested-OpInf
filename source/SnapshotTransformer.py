# credit to Shane McQuarrie

import os
import h5py
import numpy as np
import scipy.linalg as la

# the following class is just copied from Shane's code, somehow it breaks when I try to import it from his library to
# give it a child
class SnapshotTransformer:
    """Process snapshots by centering and/or scaling (in that order).

    Parameters
    ----------
    center : bool
        If True, shift the snapshots by the mean training snapshot.
    scaling : str or None
        If given, scale (non-dimensionalize) the centered snapshot entries.
        * 'standard': standardize to zero mean and unit standard deviation.
        * 'minmax': minmax scaling to [0, 1].
        * 'minmaxsym': minmax scaling to [-1, 1].
        * 'maxabs': maximum absolute scaling to [-1, 1] (no shift).
        * 'maxabssym': maximum absolute scaling to [-1, 1] (mean shift).
    verbose : bool
        If True, print information upon learning a transformation.

    Attributes
    ----------
    mean_ : (n,) ndarray
        Mean training snapshot. Only recorded if center = True.
    scale_ : float
        Multiplicative factor of scaling (the a of q -> aq + b).
        Only recorded if scaling != None.
    shift_ : float
        Additive factor of scaling (the b of q -> aq + b).
        Only recorded if scaling != None.

    Notes
    -----
    Snapshot centering (center=True):
        Q' = Q - mean(Q, axis=1);
        Guarantees mean(Q', axis=1) = [0, ..., 0].
    Standard scaling (scaling='standard'):
        Q' = (Q - mean(Q)) / std(Q);
        Guarantees mean(Q') = 0, std(Q') = 1.
    Min-max scaling (scaling='minmax'):
        Q' = (Q - min(Q))/(max(Q) - min(Q));
        Guarantees min(Q') = 0, max(Q') = 1.
    Symmetric min-max scaling (scaling='minmaxsym'):
        Q' = (Q - min(Q))*2/(max(Q) - min(Q)) - 1
        Guarantees min(Q') = -1, max(Q') = 1.
    Maximum absolute scaling (scaling='maxabs'):
        Q' = Q / max(abs(Q));
        Guarantees mean(Q') = mean(Q) / max(abs(Q)), max(abs(Q')) = 1.
    Min-max absolute scaling (scaling='maxabssym'):
        Q' = (Q - mean(Q)) / max(abs(Q - mean(Q)));
        Guarantees mean(Q') = 0, max(abs(Q')) = 1.
    """
    _VALID_SCALINGS = {
        "standard",
        "minmax",
        "minmaxsym",
        "maxabs",
        "maxabssym",
    }

    _table_header = "    |     min    |    mean    |     max    |    std\n"
    _table_header += "----|------------|------------|------------|------------"

    def __init__(self, center=False, scaling=None, verbose=False):
        """Set transformation hyperparameters."""
        self.center = center
        self.scaling = scaling
        self.verbose = verbose

    def _clear(self):
        """Delete all learned attributes."""
        for attr in ("mean_", "scale_", "shift_"):
            if hasattr(self, attr):
                delattr(self, attr)

    # Properties --------------------------------------------------------------
    @property
    def center(self):
        """Snapshot mean-centering directive (bool)."""
        return self.__center

    @center.setter
    def center(self, ctr):
        """Set the centering directive, resetting the transformation."""
        if ctr not in (True, False):
            raise TypeError("'center' must be True or False")
        self._clear()
        self.__center = ctr

    @property
    def scaling(self):
        """Entrywise scaling (non-dimensionalization) directive.
        * None: no scaling.
        * 'standard': standardize to zero mean and unit standard deviation.
        * 'minmax': minmax scaling to [0, 1].
        * 'minmaxsym': minmax scaling to [-1, 1].
        * 'maxabs': maximum absolute scaling to [-1, 1] (no shift).
        * 'maxabssym': maximum absolute scaling to [-1, 1] (mean shift).
        """
        return self.__scaling

    @scaling.setter
    def scaling(self, scl):
        """Set the scaling strategy, resetting the transformation."""
        if scl is None:
            self._clear()
            self.__scaling = scl
            return
        if not isinstance(scl, str):
            raise TypeError("'scaling' must be of type 'str'")
        if scl not in self._VALID_SCALINGS:
            opts = ", ".join([f"'{v}'" for v in self._VALID_SCALINGS])
            raise ValueError(f"invalid scaling '{scl}'; "
                             f"valid options are {opts}")
        self._clear()
        self.__scaling = scl

    @property
    def verbose(self):
        """If True, print information about upon learning a transformation."""
        return self.__verbose

    @verbose.setter
    def verbose(self, vbs):
        self.__verbose = bool(vbs)

    def __eq__(self, other):
        """Test two SnapshotTransformers for equality."""
        if not isinstance(other, self.__class__):
            return False
        for attr in ("center", "scaling"):
            if getattr(self, attr) != getattr(other, attr):
                return False
        if self.center and hasattr(self, "mean_"):
            if not hasattr(other, "mean_"):
                return False
            if not np.all(self.mean_ == other.mean_):
                return False
        if self.scaling and hasattr(self, "scale_"):
            for attr in ("scale_", "shift_"):
                if not hasattr(other, attr):
                    return False
                if getattr(self, attr) != getattr(other, attr):
                    return False
        return True

    # Printing ----------------------------------------------------------------
    def __str__(self):
        """String representation: scaling type + centering bool."""
        out = ["Snapshot transformer"]
        if self.center:
            out.append("with mean-snapshot centering")
            if self.scaling:
                out.append(f"and '{self.scaling}' scaling")
        elif self.scaling:
            out.append(f"with '{self.scaling}' scaling")
        if not self._is_trained():
            out.append("(call fit_transform() to train)")
        return ' '.join(out)

    @staticmethod
    def _statistics_report(Q):
        """Return a string of basis statistics about a data set."""
        return " | ".join([f"{f(Q):>10.3e}"
                           for f in (np.min, np.mean, np.max, np.std)])

    # Persistence -------------------------------------------------------------
    def save(self, savefile, overwrite=False):
        """Save the current transformer to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the transformer in.
        overwrite : bool
            If True, overwrite the file if it already exists. If False
            (default), raise a FileExistsError if the file already exists.
        """
        # Ensure the file is saved in HDF5 format.
        if not savefile.endswith(".h5"):
            savefile += ".h5"

        # Prevent overwriting and existing file on accident.
        if os.path.isfile(savefile) and not overwrite:
            raise FileExistsError(f"{savefile} (use overwrite=True to ignore)")

        with h5py.File(savefile, 'w') as hf:
            # Store transformation hyperparameter metadata.
            meta = hf.create_dataset("meta", shape=(0,))
            meta.attrs["center"] = self.center
            meta.attrs["scaling"] = self.scaling if self.scaling else False
            meta.attrs["verbose"] = self.verbose

            # Store learned transformation parameters.
            if self.center and hasattr(self, "mean_"):
                hf.create_dataset("transformation/mean_", data=self.mean_)
            if self.scaling and hasattr(self, "scale_"):
                hf.create_dataset("transformation/scale_", data=[self.scale_])
                hf.create_dataset("transformation/shift_", data=[self.shift_])

    @classmethod
    def load(cls, loadfile):
        """Load a SnapshotTransformer from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the transformer was stored (via save()).

        Returns
        -------
        SnapshotTransformer
        """
        with h5py.File(loadfile, 'r') as hf:
            # Load transformation hyperparameters.
            if "meta" not in hf:
                raise ValueError("invalid save format (meta/ not found)")
            scl = hf["meta"].attrs["scaling"]
            transformer = cls(center=hf["meta"].attrs["center"],
                              scaling=scl if scl else None,
                              verbose=hf["meta"].attrs["verbose"])

            # Load learned transformation parameters.
            if transformer.center and "transformation/mean_" in hf:
                transformer.mean_ = hf["transformation/mean_"][:]
            if transformer.scaling and "transformation/scale_" in hf:
                transformer.scale_ = hf["transformation/scale_"][0]
                transformer.shift_ = hf["transformation/shift_"][0]

            return transformer

    # Main routines -----------------------------------------------------------
    def _is_trained(self):
        """Return True if transform() and inverse_transform() are ready."""
        if self.center and not hasattr(self, "mean_"):
            return False
        if self.scaling and any(not hasattr(self, attr)
                                for attr in ("scale_", "shift_")):
            return False
        return True

    def fit_transform(self, states, inplace=False):
        """Learn and apply the transformation.

        Parameters
        ----------
        states : (n,k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n.
        inplace : bool
            If True, overwrite the input data during transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n,k) ndarray
            Matrix of k transformed n-dimensional snapshots.
        """

        Y = states if inplace else states.copy()

        # Record statistics of the training data.
        if self.verbose:
            report = ["No transformation learned"]
            report.append(self._table_header)
            report.append(f"Q   | {self._statistics_report(Y)}")

        # Center the snapshots by the mean training snapshot.
        if self.center:
            self.mean_ = np.mean(Y, axis=1)
            Y -= self.mean_.reshape((-1, 1))

            if self.verbose:
                report[0] = "Learned mean centering Q -> Q'"
                report.append(f"Q'  | {self._statistics_report(Y)}")

        # Scale (non-dimensionalize) the centered snapshot entries.
        if self.scaling:
            # Standard: Q' = (Q - mu)/sigma
            if self.scaling == "standard":
                mu = np.mean(Y)
                sigma = np.std(Y)
                self.scale_ = 1/sigma
                self.shift_ = -mu*self.scale_

            # Min-max: Q' = (Q - min(Q))/(max(Q) - min(Q))
            elif self.scaling == "minmax":
                Ymin = np.min(Y)
                Ymax = np.max(Y)
                self.scale_ = 1/(Ymax - Ymin)
                self.shift_ = -Ymin*self.scale_

            # Symmetric min-max: Q' = (Q - min(Q))*2/(max(Q) - min(Q)) - 1
            elif self.scaling == "minmaxsym":
                Ymin = np.min(Y)
                Ymax = np.max(Y)
                self.scale_ = 2/(Ymax - Ymin)
                self.shift_ = -Ymin*self.scale_ - 1

            # MaxAbs: Q' = Q / max(abs(Q))
            elif self.scaling == "maxabs":
                self.scale_ = 1/np.max(np.abs(Y))
                self.shift_ = 0

            # maxabssym: Q' = (Q - mean(Q)) / max(abs(Q - mean(Q)))
            elif self.scaling == "maxabssym":
                mu = np.mean(Y)
                Y -= mu
                self.scale_ = 1/np.max(np.abs(Y))
                self.shift_ = -mu*self.scale_
                Y += mu

            else:                               # pragma nocover
                raise RuntimeError(f"invalid scaling '{self.scaling}'")

            Y *= self.scale_
            Y += self.shift_

            if self.verbose:
                if self.center:
                    report[0] += f" and {self.scaling} scaling Q' -> Q''"
                else:
                    report[0] = f"Learned {self.scaling} scaling Q -> Q''"
                report.append(f"Q'' | {self._statistics_report(Y)}")

        if self.verbose:
            print('\n'.join(report) + '\n')

        return Y

    def transform(self, states, inplace=False):
        """Apply the learned transformation.

        Parameters
        ----------
        states : (n,k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n.
        inplace : bool
            If True, overwrite the input data during transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n,k) ndarray
            Matrix of k transformed n-dimensional snapshots.
        """
        if not self._is_trained():
            raise AttributeError("transformer not trained "
                                 "(call fit_transform())")

        Y = states if inplace else states.copy()

        # Center the snapshots by the mean training snapshot.
        if self.center is True:
            Y -= self.mean_.reshape((-1, 1))

        # Scale (non-dimensionalize) the centered snapshot entries.
        if self.scaling is not None:
            Y *= self.scale_
            Y += self.shift_

        return Y

    def inverse_transform(self, states_transformed, inplace=False):
        """Apply the inverse of the learned transformation.

        Parameters
        ----------
        states_transformed : (n,k) ndarray
            Matrix of k transformed n-dimensional snapshots.
        inplace : bool
            If True, overwrite the input data during inverse transformation.
            If False, create a copy of the data to untransform.

        Returns
        -------
        states: (n,k) ndarray
            Matrix of k untransformed n-dimensional snapshots.
        """
        if not self._is_trained():
            raise AttributeError("transformer not trained "
                                 "(call fit_transform())")

        Y = states_transformed if inplace else states_transformed.copy()

        # Unscale (re-dimensionalize) the data.
        if self.scaling:
            Y -= self.shift_
            Y /= self.scale_

        # Uncenter the unscaled snapshots.
        if self.center:
            Y += self.mean_.reshape((-1, 1))

        return Y

class SnapshotTransformerExtension(SnapshotTransformer):

    _VALID_SCALINGS = {
        "standard",
        "minmax",
        "minmaxsym",
        "maxabs",
        "maxabssym",
        "maxnorm"
    }

    def fit_transform(self, states, inplace=False):
        """Learn and apply the transformation.

        Parameters
        ----------
        states : (n,k) ndarray
            Matrix of k snapshots. Each column is a snapshot of dimension n.
        inplace : bool
            If True, overwrite the input data during transformation.
            If False, create a copy of the data to transform.

        Returns
        -------
        states_transformed: (n,k) ndarray
            Matrix of k transformed n-dimensional snapshots.
        """

        if self.scaling not in ["maxnorm"]:
            return super().fit_transform(states=states, inplace=inplace)

        print("here")

        Y = states if inplace else states.copy()

        # Record statistics of the training data.
        if self.verbose:
            report = ["No transformation learned"]
            report.append(self._table_header)
            report.append(f"Q   | {self._statistics_report(Y)}")

        # Center the snapshots by the mean training snapshot.
        if self.center:
            self.mean_ = np.mean(Y, axis=1)
            Y -= self.mean_.reshape((-1, 1))

            if self.verbose:
                report[0] = "Learned mean centering Q -> Q'"
                report.append(f"Q'  | {self._statistics_report(Y)}")

        # Scale (non-dimensionalize) the centered snapshot entries.
        if self.scaling:

            if self.scaling == "maxnorm":

                print("here")

                self.scale_ = 1 / np.max(la.norm(Y, axis=0))
                self.shift_ = 0

            else:  # pragma nocover
                raise RuntimeError(f"invalid scaling '{self.scaling}'")

            Y *= self.scale_
            Y += self.shift_

            if self.verbose:
                if self.center:
                    report[0] += f" and {self.scaling} scaling Q' -> Q''"
                else:
                    report[0] = f"Learned {self.scaling} scaling Q -> Q''"
                report.append(f"Q'' | {self._statistics_report(Y)}")

        if self.verbose:
            print('\n'.join(report) + '\n')

        return Y



