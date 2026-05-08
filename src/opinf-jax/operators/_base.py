import os
import abc
import copy
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import equinox as eqx
import matplotlib.pyplot as plt

from ...opinf import errors, utils


class InputMixin(eqx.Module):
    r"""Mixin for operators whose ``apply()`` method acts on the input
    :math:`\u`.

    Operators that do not inherit from this Mixin do not have an
    ``input_dimension`` attribute, which indicates :math:`m = 0`.
    """
    @property
    @abc.abstractmethod
    def input_dimension(self) -> int:
        r"""Dimension :math:`m` of the input :math:`\u` that the operator
        acts on.
        """
        raise NotImplementedError # pragma: no cover
    
def has_inputs(obj) -> bool:
    r"""Return ``True`` if ``obj`` is an operator object whose ``apply()``
    method acts on the ``input_`` argument, i.e.,
    :math:`\Ophat_{\ell}(\qhat,\u)` depends on :math:`\u`.
    """
    return isinstance(obj, InputMixin)

class OperatorTemplate(eqx.Module):
    r"""Template for general operators :math:`\Ophat_{\ell}(\qhat,\u).`

    In this package, an "operator" is a function
    :math:`\Ophat_{\ell}: \RR^r \times \RR^m \to \RR^r` that acts on a state
    vector :math:`\qhat\in\RR^r` and (optionally) an input vector
    :math:`\u\in\RR^m`.

    Models are defined as the sum of several operators,
    for example, an :class:`opinf.models.ContinuousModel` object represents a
    system of ordinary differential equations:

    .. math::
       \ddt\qhat(t)
       = \sum_{\ell=1}^{n_\textrm{terms}}\Ophat_{\ell}(\qhat(t),\u(t)).

    Notes
    -----
    This class can be used for custom nonparametric model terms that are not
    learnable with Operator Inference.
    For parametric model terms, see :class:`ParametricOperatorTemplate`.
    For model terms that can be learned with Operator Inference, see
    :class:`OpInfOperator` or :class:`ParametricOpInfOperator`.
    """

    # Properties --------------------------------------------------------------
    @property
    @abc.abstractmethod
    def state_dimension(self) -> int:
        r"""Dimension :math:`r` of the state :math:`\qhat` that the operator
        acts on.
        """
        raise NotImplementedError  # pragma: no cover

    def __str__(self) -> str:
        """String representation: class name + dimensions."""
        out = [self.__class__.__name__]
        out.append(f"state_dimension: {self.state_dimension}")
        if has_inputs(self):
            out.append(f"input_dimension: {self.input_dimension}")
        return "\n  ".join(out)

    def __repr__(self) -> str:
        return utils.str2repr(self)

    @staticmethod
    def _str(statestr, inputstr=None):
        """String representation of the operator, used when printing out the
        structure of a model.

        Parameters
        ----------
        statestr : str
            String representation of the state, e.g., ``"q(t)"`` or ``"q_j"``.
        inputstr : str
            String representation of the input, e.g., ``"u(t)"`` or ``"u_j"``.

        Returns
        -------
        opstr : str
            String representation of the operator acting on the state/input,
            e.g., ``"Aq(t)"`` or ``"Bu(t)"`` or ``"H[q(t) ⊗ q(t)]"``.
        """
        return f"f({statestr}, {inputstr})"  # pragma: no cover

    # Evaluation --------------------------------------------------------------
    @abc.abstractmethod
    def apply(self, state: jax.Array, input_=None) -> jax.Array:
        """Apply the operator mapping to the given state / input.

        Parameters
        ----------
        state : (r,) or (r, k) ndarray
            State vector or matrix of state vectors.
        input_ : (m,) or (m, k) ndarray or None
            Input vector or matrix of input vectors.

        Returns
        -------
        out : (r,) or (r, k) ndarray
            Application of the operator to the state / input, with the same
            number of dimensions as ``state`` and (if provided) ``input_``.
        """
        raise NotImplementedError  # pragma: no cover

    def jacobian(self, state: jax.Array, input_=None) -> jax.Array:
        r"""Construct the state Jacobian of the operator.

        If :math:`[\![\q]\!]_{i}` denotes the :math:`i`-th entry of a vector
        :math:`\q`, then the :math:`(i,j)`-th entry of the state Jacobian is
        given by

        .. math::
           [\![\ddqhat\Ophat(\qhat,\u)]\!]_{i,j}
           = \frac{\partial}{\partial[\![\qhat]\!]_j}
           [\![\Ophat(\qhat,\u)]\!]_i.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or float or None
            Input vector.

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian.
        """
        raise NotImplementedError  # pragma: no cover

    # Dimensionality reduction ------------------------------------------------
    def galerkin(self, Vr: jax.Array, Wr=None):
        r"""Get the (Petrov-)Galerkin projection of this operator.

        Consider an operator :math:`\Op(\q,\u)`, where :math:`\q\in\RR^n`
        is the state and :math:`\u\in\RR^m` is the input.
        Given a *trial basis* :math:`\Vr\in\RR^{n\times r}` and a *test basis*
        :math:`\Wr\in\RR^{n\times r}`, the Petrov-Galerkin projection of
        :math:`\Op` is the operator :math:`\Ophat:\RR^r\times\RR^m\to\RR^r`
        defined by

        .. math::
           \Ophat(\qhat, \u) = (\Wr\trp\Vr)^{-1}\Wr\trp\Op(\Vr\qhat, \u)

        where :math:`\qhat\in\RR^n` approximates the original state via
        :math:`\q \approx \Vr\qhat`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space :math:`\Vr`.
        Wr : (n, r) ndarray or None
            Basis for the test space :math:`\Wr`.
            If ``None`` (default), use ``Vr`` as the test basis.

        Returns
        -------
        op : :class:`OperatorTemplate`
            New operator object whose ``state_dimension``
            attribute equals ``r``. If this operator acts on inputs, the
            ``input_dimension`` attribute of the new operator should be
            ``self.input_dimension``.
        """
        raise NotImplementedError  # pragma: no cover

    # Model persistence -------------------------------------------------------
    def copy(self):
        """Return a copy of the operator using :func:`copy.deepcopy()`."""
        return copy.deepcopy(self)  # pragma: no cover

    def save(self, savefile: str, overwrite: bool = False) -> None:
        """Save the operator to an HDF5 file.

        Parameters
        ----------
        savefile : str
            Path of the file to save the basis in.
        overwrite : bool
            If ``True``, overwrite the file if it already exists. If ``False``
            (default), raise a ``FileExistsError`` if the file already exists.
        """
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def load(cls, loadfile: str):
        """Load an operator from an HDF5 file.

        Parameters
        ----------
        loadfile : str
            Path to the file where the operator was stored via :meth:`save()`.
        """
        raise NotImplementedError  # pragma: no cover

    # Verification ------------------------------------------------------------
    def verify(
        self,
        plot: bool = False,
        *,
        k: int = 10,
        fdifftol: float = 1e-5,
        ntests: int = 4,
    ) -> None:  # pragma: no cover
        """Verify consistency between dimension properties and required
        methods.

        This method verifies :meth:`apply()` and, if implemented,
        :meth:`jacobian()`, :meth:`galerkin()`, :meth:`copy()`,
        :meth:`save()`, and :meth:`load()`.

        Parameters
        ----------
        plot : bool
            If ``True``, plot the relative errors of the finite difference
            check for :meth:`jacobian()` as a function of the perturbation
            size.
            If ``False`` (default), print a report of the relative errors.
            Nothing is plotted or printed if :meth:`jacobian()` is not
            implemented.

        Notes
        -----
        This method does **not** verify the correctness of :meth:`apply()`,
        only that it returns an output with the expected shape. However,
        if :meth:`jacobian()` is implemented, a finite difference check is
        applied to check that :meth:`apply()` and :meth:`jacobian()` are
        consistent.
        """
        # Verify dimensions exist and are valid.
        if (
            not isinstance((r := self.state_dimension), int) or r <= 0
        ):  # pragma: no cover
            raise errors.VerificationError(
                "state_dimension must be a positive integer "
                f"(current value: {repr(r)}, of type '{type(r).__name__}')"
            )

        if hasinputs := has_inputs(self):
            if (
                not isinstance((m := self.input_dimension), int) or m <= 0
            ):  # pragma: no cover
                raise errors.VerificationError(
                    "input_dimension must be a positive integer "
                    f"(current value: {repr(m)}, of type '{type(r).__name__}')"
                )
        else:
            m = 0

        # Verify apply() - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Q = np.random.random((r, k))
        q = Q[:, 0]
        U, u = None, None
        if hasinputs:
            U = np.random.random((m, k))
            u = U[:, 0]

        out = self.apply(q, u)
        if not isinstance(out, jax.Array) or out.shape != (
            r,
        ):  # pragma: no cover
            _message = [
                "apply(q, u) must return array of shape (state_dimension,)",
                "when q.shape = (state_dimension,)",
                "and u = None",
            ]
            if hasinputs:
                _message[-1] = "and u.shape = (input_dimension,)"
            raise errors.VerificationError(" ".join(_message))

        out = self.apply(Q, U)
        if not isinstance(out, jax.Array) or out.shape != (
            r,
            k,
        ):  # pragma: no cover
            _message = [
                "apply(Q, U) must return array of shape (state_dimension, k)",
                "when Q.shape = (state_dimension, k)",
                "and U = None",
            ]
            if hasinputs:
                _message[-1] = "and U.shape = (input_dimension, k)"
            raise errors.VerificationError(" ".join(_message))

        # Report successes.
        def _report_isconsistent(method):
            _message = f"{method} is consistent with state_dimension"
            if hasinputs:
                _message += " and input_dimension"
            print(_message)

        _report_isconsistent("apply()")

        # Verify jacobian() - - - - - - - - - - - - - - - - - - - - - - - - - -
        def _gradient(f, x, h=1e-8):
            """Estimate the Jacobian of f:R^n -> R^m at x using perturbations
            of magnitude h.
            """
            E = jnp.eye((n := x.size))
            return jnp.array([(f(x + h * E[i]) - f(x)) / h for i in range(n)]).T

        def _finite_difference_check(f, df, x, hs=None):
            """Compare analytical and numerical derivatives."""
            dfx = df(x)
            if hs is None:
                hs = jnp.logspace(-10, -1, 10)[::-1]
            return hs, jnp.array(
                [jla.norm(_gradient(f, x, h) - dfx) / jla.norm(dfx) for h in hs]
            )

        try:
            out = self.jacobian(q, u)
        except NotImplementedError:  # pragma: no cover
            print("jacobian() not implemented")
        else:
            if jnp.isscalar(out) and out == 0:  # pragma: no cover
                print("jacobian() = 0")
            elif not isinstance(out, jax.Array) or out.shape != (
                r,
                r,
            ):  # pragma: no cover
                _message = [
                    "jacobian(q, u) must return array",
                    "of shape (state_dimension, state_dimension)",
                    "when q.shape = (state_dimension,)",
                    "and u = None",
                ]
                if hasinputs:
                    _message[-1] = "and u.shape = (input_dimension,)"
                raise errors.VerificationError(" ".join(_message))
            else:
                _report_isconsistent("jacobian()")

                # Finite difference check.
                hs, diffs = _finite_difference_check(
                    lambda x: self.apply(x, u),
                    lambda x: self.jacobian(x, u),
                    np.random.standard_normal(r),
                )
                if plot:  # pragma: no cover
                    plt.loglog(hs, diffs, ".-", markersize=5, linewidth=0.5)
                else:
                    print(
                        "jacobian() finite difference relative errors",
                        "  ------------------------------------------",
                        sep="\n",
                    )
                    for h, err in zip(hs, diffs):
                        print(f"  h = {h:.2e}\terror = {err:.4e}")
                if jnp.min(diffs) > fdifftol:  # pragma: no cover
                    raise errors.VerificationError(
                        "jacobian() finite difference check failed"
                    )

        # Verify galerkin() - - - - - - - - - - - - - - - - - - - - - - - - - -
        def _orth(n, k):
            """Get an n x k matrix with orthonormal columns."""
            return jla.qr(np.random.standard_normal((n, k)), mode="economic")[0]

        if r > 1:
            rnew = r // 2
            Vr = _orth(r, rnew)
            Wr = _orth(r, rnew)
            try:
                out = self.galerkin(Vr, Wr)
            except NotImplementedError:  # pragma: no cover
                print("galerkin() not implemented")
            else:
                if not isinstance(out, OperatorTemplate):  # pragma: no cover
                    raise errors.VerificationError(
                        "galerkin() must return object "
                        "whose class inherits from OperatorTemplate"
                    )
                if out.state_dimension != rnew:  # pragma: no cover
                    raise errors.VerificationError(
                        "galerkin(Vr, Wr).state_dimension != Vr.shape[1]"
                    )
                if hasinputs and out.input_dimension != m:  # pragma: no cover
                    raise errors.VerificationError(
                        "self.galerkin(Vr, Wr).input_dimension "
                        "!= self.input_dimension"
                    )
                WrTVr_LU = jla.lu_factor(Wr.T @ Vr)
                for _ in range(ntests):
                    qr = np.random.random(rnew)
                    full = jla.lu_solve(WrTVr_LU, Wr.T @ self.apply(Vr @ qr, u))
                    reduced = out.apply(qr, u)
                    if not jnp.allclose(reduced, full):  # pragma: no cover
                        raise errors.VerificationError(
                            "op2.apply(qr, u) != "
                            "inv(Wr.T @ Vr) @ Wr.T @ self.apply(Vr @ qr, u) "
                            "where op2 = self.galerkin(Vr, Wr)"
                        )
                out = self.galerkin(Vr)
                for _ in range(ntests):
                    qr = np.random.random(rnew)
                    full = Vr.T @ self.apply(Vr @ qr, u)
                    reduced = out.apply(qr, u)
                    if not jnp.allclose(reduced, full):  # pragma: no cover
                        raise errors.VerificationError(
                            "op2.apply(qr, u) != "
                            "Vr.T @ self.apply(Vr @ qr, u) "
                            "where op2 = self.galerkin(Vr) and Vr.T @ Vr = I"
                        )
                print("galerkin() is consistent with apply()")
        else:  # pragma: no cover
            print("cannot test galerkin() when state_dimension = 1")

        # Verify copy() - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        out = self.copy()
        if out is self:  # pragma: no cover
            raise errors.VerificationError("self.copy() is self")
        if out.__class__ is not self.__class__:  # pragma: no cover
            raise errors.VerificationError(
                "type(self.copy()) is not type(self)"
            )
        if out.state_dimension != r:  # pragma: no cover
            raise errors.VerificationError(
                "self.copy().state_dimension != self.state_dimension"
            )
        if hasinputs and out.input_dimension != m:  # pragma: no cover
            raise errors.VerificationError(
                "self.copy().input_dimension != self.input_dimension"
            )
        _report_isconsistent("copy()")
        for _ in range(ntests):
            q = np.random.random(r)
            if not jnp.allclose(
                out.apply(q, u), self.apply(q, u)
            ):  # pragma: no cover
                raise errors.VerificationError(
                    "self.copy().apply() not consistent with self.apply()"
                )
        print("copy() preserves the results of apply()")

        # Verify save()/load() - - - - - - - - - - - - - - - - - - - - - - - -
        tempfile = "_operatorverification.h5"
        try:
            self.save(tempfile)
            out = self.load(tempfile)
        except NotImplementedError:  # pragma: no cover
            print("save() and/or load() not implemented")
        else:
            if out.__class__ is not self.__class__:  # pragma: no cover
                raise errors.VerificationError(
                    "save()/load() does not preserve object type"
                )
            if out.state_dimension != r:  # pragma: no cover
                raise errors.VerificationError(
                    "save()/load() does not preserve state_dimension"
                )
            if hasinputs and out.input_dimension != m:  # pragma: no cover
                raise errors.VerificationError(
                    "save()/load() does not preserve input_dimension"
                )
            _report_isconsistent("save()/load()")
            for _ in range(ntests):
                q = np.random.random(r)
                if not jnp.allclose(
                    out.apply(q, u), self.apply(q, u)
                ):  # pragma: no cover
                    raise errors.VerificationError(
                        "save()/load() does not preserve the result of apply()"
                    )
            print("save()/load() preserves the results of apply()")
        finally:
            if os.path.isfile(tempfile):  # pragma: no cover
                os.remove(tempfile)


def is_nonparametric(obj) -> bool:
    """Return ``True`` if ``obj`` is a nonparametric operator object."""
    return isinstance(obj, OperatorTemplate)

