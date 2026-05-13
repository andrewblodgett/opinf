import itertools
import math

import jax
import jax.numpy as jnp
import jax.scipy.linalg as la
import scipy.special as special

from .. import utils
from ._base import OpInfOperator

class ConstantOperator(OpInfOperator):

    def __init__(self, entries: jax.Array):
        if jnp.isscalar(entries):
            entries = jnp.atleast_1d(entries)
        self._validate_entries(entries)
        # Ensure that the operator is one-dimensional.
        if entries.ndim != 1:
            if entries.ndim == 2 and 1 in entries.shape:
                entries = jnp.ravel(entries)
            else:
                raise ValueError(
                    "ConstantOperator entries must be one-dimensional"
                )
        self._entries = entries

    @staticmethod
    def _str(statestr=None, inputstr=None):
        return "c"

    @property
    def entries(self):
        r"""Operator vector :math:`\chat`."""
        return OpInfOperator.entries.fget(self)

    @property
    def shape(self):
        r"""Shape :math:`(r,)` of the operator vector :math:`\chat`."""
        return OpInfOperator.shape.fget(self)
    
    @utils.requires("entries")
    def apply(self, state=None, input_=None):
        r"""Apply the operator to the given state / input:
        :math:`\Ophat_{\ell}(\qhat,\u) = \chat`.

        Parameters
        ----------
        state : (r,) ndarray or None
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        out : (r,) ndarray
            :math:`\chat`.
        """
        if self.entries.shape[0] == 1:
            if state is None or jnp.isscalar(state):  # r = k = 1.
                return self.entries[0]
            return jnp.full_like(state, self.entries[0])  # r = 1, k > 1.
        # if state is None or np.ndim(state) == 1:
        #     return self.entries
        if jnp.ndim(state) == 2:  # r, k > 1.
            return jnp.outer(self.entries, jnp.ones(state.shape[-1]))
        return self.entries 
    
    @utils.requires("entries")
    def galerkin(self, Vr, Wr=None):
        r"""Return the Galerkin projection of the operator,
        :math:`\chat = (\Wr\trp\Vr)^{-1}\Wr\trp\c`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        projected : :class:`opinf.operators.ConstantOperator`
            Projected operator.
        """
        return self._galerkin(Vr, Wr, lambda c, V: c)
    
    @staticmethod
    def datablock(states, inputs=None):
        r"""Return the data matrix block corresponding to the operator,
        a row vector of ones.

        Since :math:`\Ophat_\ell(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat,\u)`
        with :math:`\Ohat_{\ell} = \chat` and :math:`\d_{\ell}(\qhat,\u) = 1`,
        the data block is

        .. math::
           \D\trp
           = \left[\begin{array}{ccc}
           \d_{\ell}(\qhat_0,\u_0)
           & \cdots &
           \d_{\ell}(\qhat_{k-1},\u_{k-1})
           \end{array}\right]
           = \left[\begin{array}{ccc}
           1 & \cdots & 1
           \end{array}\right]
           \in \RR^{1 \times k}.

        Parameters
        ----------
        states : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        inputs : (m, k) or (k,) ndarray or None
            Input vectors (not used).

        Returns
        -------
        block : (1, k) ndarray
            Row vector of ones.
        """
        return jnp.ones((1, jnp.atleast_1d(states).shape[-1]))
    
    @staticmethod
    def operator_dimension(r=None, m=None):
        r"""Column dimension of the operator vector (always 1).

        Parameters
        ----------
        r : int
            State dimension.
        m : int or None
            Input dimension.
        """
        return 1
    
class LinearOperator(OpInfOperator):
    def __init__(self, entries):
        if jnp.isscalar(entries):
            entries = jnp.atleast_2d(entries)
        self._validate_entries(entries)
        if entries.ndim != 2:
            raise ValueError("LinearOperator entries must be two-dimensional")
        if entries.shape[0] != entries.shape[1]:
            raise ValueError("LinearOperator entries must be square (r x r)")
        self._entries = entries

    @staticmethod
    def _str(statestr, inputstr=None):
        return f"A{statestr}"

    @property
    def entries(self):
        r"""Operator matrix :math:`\Ahat`."""
        return OpInfOperator.entries.fget(self)
    
    @property
    def shape(self):
        r"""Shape :math:`(r, r)` of the operator matrix :math:`\Ahat`."""
        return OpInfOperator.shape.fget(self)
    
    def apply(self, state, input_=None):
        r"""Apply the operator to the given state / input:
        :math:`\Ophat_{\ell}(\qhat,\u) = \Ahat\qhat`.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        out : (r,) ndarray
            Application :math:`\Ahat\qhat`.
        """
        if self.entries.shape[0] == 1:
            return self.entries[0, 0] * state  # r = 1.
        return self.entries @ state  # r > 1.
    
    def jacobian(self, state=None, input_=None):
        r"""Construct the state Jacobian of the operator:
        :math:`\ddqhat\Ophat_{\ell}(\qhat,\u)=\Ahat`.

        Parameters
        ----------
        state : (r,) ndarray or None
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian :math:`\Ahat`.
        """
        return self.entries
    
    def galerkin(self, Vr, Wr=None):
        r"""Return the Galerkin projection of the operator,
        :math:`\Ahat = (\Wr\trp\Vr)^{-1}\Wr\trp\A\Vr`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        projected : :class:`opinf.operators.LinearOperator`
            Projected operator.
        """
        return self._galerkin(Vr, Wr, lambda A, V: A @ V)
    
    @staticmethod
    def datablock(states, inputs=None):
        r"""Return the data matrix block corresponding to the operator,
        the ``states``.

        Since :math:`\Ophat_\ell(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat,\u)`
        with :math:`\Ohat_{\ell} = \Ahat` and
        :math:`\d_{\ell}(\qhat,\u) = \qhat`, the data block is

        .. math::
           \D\trp
           = \left[\begin{array}{ccc}
           \d_{\ell}(\qhat_0,\u_0)
           & \cdots &
           \d_{\ell}(\qhat_{k-1},\u_{k-1})
           \end{array}\right]
           = \left[\begin{array}{ccc}
           \qhat_0 & \cdots & \qhat_{k-1}
           \end{array}\right]
           \in \RR^{r \times k}.

        Parameters
        ----------
        states : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        inputs : (m, k) or (k,) ndarray or None
            Input vectors (not used).

        Returns
        -------
        state : (r, k) ndarray
            State vectors. Each column is a single state vector.
        """
        return jnp.atleast_2d(states)

    @staticmethod
    def operator_dimension(r, m=None):
        r"""Column dimension :math:`r` of the operator matrix :math:`\Ahat`.

        Parameters
        ----------
        r : int
            State dimension.
        m : int or None
            Input dimension.
        """
        return r
    
class QuadraticOperator(OpInfOperator):
    _mask: jax.Array
    _prejac: jax.Array

    def __init__(self, entries):
        if jnp.isscalar(entries) or jnp.shape(entries) == (1,):
            entries = jnp.atleast_2d(entries)
        self._validate_entries(entries)
        # Ensure that the operator has valid dimensions.
        if entries.ndim == 3 and len(set(entries.shape)) == 1:
            # Reshape (r x r x r) tensor.
            entries = entries.reshape((entries.shape[0], -1))
        if entries.ndim != 2:
            raise ValueError(
                "QuadraticOperator entries must be two-dimensional"
            )
        r, r2 = entries.shape
        if r2 == r**2:
            entries = self.compress_entries(entries)
        elif r2 != self.operator_dimension(r):
            raise ValueError("invalid QuadraticOperator entries dimensions")
        
        self._entries = entries
        self._mask = self.ckron_indices(r)
        self._prejac = self._precompute_jacobian_jit()


    def _precompute_jacobian_jit(self):
        """Compute (just in time) the pre-Jacobian tensor Jt such that
        Jt @ q = jacobian(q).
        """
        r = self.entries.shape[0]
        Ht = self.expand_entries(self.entries).reshape((r, r, r))
        return Ht + Ht.transpose(0, 2, 1)

    @staticmethod
    def _str(statestr, inputstr=None):
        return f"H[{statestr} ⊗ {statestr}]"
    
    @property
    def entries(self):
        r"""Internal representation :math:`\tilde{\H}` of the operator
        matrix :math:`\Hhat`.
        """
        return OpInfOperator.entries.fget(self)

    @property
    def shape(self):
        r"""Shape :math:`(r, r(r+1)/2)` of the internal representation
        :math:`\tilde{\H}` of the operator matrix :math:`\Hhat`.
        """
        return OpInfOperator.shape.fget(self)

    @jax.jit
    def apply(self, state, input_=None):
        r"""Apply the operator to the given state / input:
        :math:`\Ophat_{\ell}(\qhat,\u) = \Hhat[\qhat\otimes\qhat]`

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        out : (r,) ndarray
            Application :math:`\Hhat[\qhat\otimes\qhat]`.
        """
        if self.entries.shape[0] == 1:
            return self.entries[0, 0] * state**2  # r = 1
        return self.entries @ jnp.prod(state[self._mask], axis=1)


    @jax.jit
    def jacobian(self, state, input_=None):  
        r"""Construct the state Jacobian of the operator:
        :math:`\ddqhat\Ophat_{\ell}(\qhat,\u)
        = \Hhat[(\I_r\otimes\qhat) + (\qhat\otimes\I_r)]`.

        Parameters
        ----------
        state : (r,) ndarray or None
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian
            :math:`\Hhat[(\I_r\otimes\qhat) + (\qhat\otimes\I_r)]`.
        """      
        return self._prejac @ jnp.atleast_1d(state)





    @staticmethod
    def datablock(states, inputs=None):
        r"""Return the data matrix block corresponding to the operator,
        the Khatri--Rao product of the state with itself:
        :math:`\Qhat\odot\Qhat` where :math:`\Qhat` is ``states``.

        Since :math:`\Ophat_\ell(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat,\u)`
        with :math:`\Ohat_{\ell} = \Hhat` and
        :math:`\d_{\ell}(\qhat,\u) = \qhat\otimes\qhat`,
        the data block should be

        .. math::
           \D\trp
           = \left[\begin{array}{ccc}
           \d_{\ell}(\qhat_0,\u_0)
           & \cdots &
           \d_{\ell}(\qhat_{k-1},\u_{k-1})
           \end{array}\right]
           = \left[\begin{array}{ccc}
           \qhat_0\otimes\qhat_0 & \cdots & \qhat_{k-1}\otimes\qhat_{k-1}
           \end{array}\right]
           \in\RR^{r^2 \times k}.

        Internally, a compressed Kronecker product :math:`\hat{\otimes}` with
        :math:`r(r+1)/2 < r^{2}` degrees of freedom is used for efficiency,
        hence the data block is actually

        .. math::
           \D\trp
           = \left[\begin{array}{ccc}
           \qhat_0\,\hat{\otimes}\,\qhat_0
           & \cdots &
           \qhat_{k-1}\,\hat{\otimes}\,\qhat_{k-1}
           \end{array}\right]
           \in\RR^{r(r+1)/2 \times k}.

        Parameters
        ----------
        states : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        inputs : (m, k) or (k,) ndarray or None
            Input vectors (not used).

        Returns
        -------
        product : (r(r+1)/2, k) ndarray
            Compressed Khatri--Rao product of ``states`` with itself.
        """
        return QuadraticOperator.ckron(jnp.atleast_2d(states))

    @staticmethod
    def operator_dimension(r, m=None):
        r"""Column dimension :math:`r(r+1)/2` of the internal representation
        :math:`\tilde{\H}` of the operator matrix :math:`\Hhat`.

        Parameters
        ----------
        r : int
            State dimension.
        m : int or None
            Input dimension.
        """
        return r * (r + 1) // 2

    @staticmethod
    def ckron(state):
        row_idx, col_idx = jnp.tril_indices(state.shape[0])
        return state[row_idx, ...] * state[col_idx, ...]
    
    @staticmethod
    def ckron_indices(r):
        """Construct a mask for efficiently computing the compressed Kronecker product."""
        row_idx, col_idx = jnp.tril_indices(r)
        return jnp.column_stack((row_idx, col_idx))

    @staticmethod
    @jax.jit
    def compress_entries(H):
        if jnp.ndim(H) == 1:
            H = jnp.atleast_2d(H)
        a, r2 = H.shape
        r = math.isqrt(r2)
        if r**2 != r2:
            raise ValueError(f"invalid shape (a, r2) = {H.shape} with r2 not a perfect square")

        Ht = H.reshape((a, r, r))

        row_idx, col_idx = jnp.tril_indices(r)

        diag_mask = (row_idx == col_idx)
        Hc = jnp.where(
            diag_mask, 
            Ht[:, row_idx, col_idx], 
            Ht[:, row_idx, col_idx] + Ht[:, col_idx, row_idx]
        )

        return Hc

    @staticmethod
    @jax.jit
    def expand_entries(Hc):
        if jnp.ndim(Hc) == 1:
            Hc = jnp.atleast_2d(Hc)
        a, b = Hc.shape
        r = (math.isqrt(1 + 8 * b) - 1) // 2
        if r * (r + 1) // 2 != b:
            raise ValueError(f"invalid shape (a, r2) = {Hc.shape} with r2 != r(r+1)/2 for any integer r")

        row_idx, col_idx = jnp.tril_indices(r)

        diag_mask = (row_idx == col_idx)
        Hc_fill = jnp.where(diag_mask, Hc, Hc / 2.0)
        Ht = jnp.zeros((a, r, r))
        Ht = Ht.at[:, row_idx, col_idx].set(Hc_fill)
        Ht = Ht.at[:, col_idx, row_idx].set(Hc_fill)

        return Ht.reshape((a, r**2))

class CubicOperator(OpInfOperator):
    r"""Cubic state operator
    :math:`\Ophat_{\ell}(\qhat,\u) = \Ghat[\qhat\otimes\qhat\otimes\qhat]`
    where :math:`\Ghat\in\RR^{r \times r^{3}}`.

    Internally, the action of the operator is computed as the product of an
    :math:`r \times r(r+1)(r+2)/6` matrix :math:`\tilde{\G}` and a compressed
    version of the triple Kronecker product
    :math:`\qhat \otimes \qhat \otimes \qhat`.

    Parameters
    ----------
    entries : (r, r^3) or (r, r(r+1)(r+2)/6) or (r, r, r, r) ndarray or None
        Operator matrix :math:`\Ghat`, its compressed representation
        :math:`\tilde{\G}`, or the equivalent symmetric 4-tensor.

    Examples
    --------
    >>> import numpy as np
    >>> G = opinf.operators.CubicOperator()
    >>> entries = np.random.random((10, 1000))  # Operator matrix.
    >>> G.set_entries(entries)
    >>> G.shape                                 # Compressed shape.
    (10, 220)
    >>> q = np.random.random(10)                # State vector.
    >>> out = G.apply(q)                        # Apply the operator to q.
    >>> np.allclose(out, entries @ np.kron(q, np.kron(q, q)))
    True
    """
    _mask: jax.Array
    _prejac: jax.Array

    def __init__(self, entries):
        if jnp.isscalar(entries) or jnp.shape(entries) == (1,):
            entries = jnp.atleast_2d(entries)
        self._validate_entries(entries)

        # Ensure that the operator has valid dimensions.
        if entries.ndim == 4 and len(set(entries.shape)) == 1:
            # Reshape (r x r x r x r) tensor.
            entries = entries.reshape((entries.shape[0], -1))
        if entries.ndim != 2:
            raise ValueError("CubicOperator entries must be two-dimensional")
        r, r3 = entries.shape
        if r3 == r**3:
            entries = self.compress_entries(entries)
        elif r3 != self.operator_dimension(r):
            raise ValueError("invalid CubicOperator entries dimensions")

        self._entries = entries
        self._mask = self.ckron_indices(r)
        self._prejac = self._precompute_jacobian_jit()

    @staticmethod
    def _str(statestr, inputstr=None):
        return f"G[{statestr} ⊗ {statestr} ⊗ {statestr}]"

    def _precompute_jacobian_jit(self):
        """Compute (just in time) the pre-Jacobian tensor Jt such that
        (Jt @ q) @ q = jacobian(q).
        """
        r = self.entries.shape[0]
        Gt = self.expand_entries(self.entries).reshape((r, r, r, r))
        return Gt + Gt.transpose(0, 2, 1, 3) + Gt.transpose(0, 3, 1, 2)

    @property
    def entries(self):
        r"""Internal representation :math:`\tilde{\G}` of the operator
        matrix :math:`\Ghat`.
        """
        return OpInfOperator.entries.fget(self)

    @property
    def shape(self):
        r"""Shape :math:`(r, r(r+1)(r+2)/6)` of the internal representation
        :math:`\tilde{\G}` of the operator matrix :math:`\Ghat`.
        """
        return OpInfOperator.shape.fget(self)

    @utils.requires("entries")
    def apply(self, state, input_=None):
        r"""Apply the operator to the given state / input:
        :math:`\Ophat_{\ell}(\qhat,\u) = \Ghat[\qhat\otimes\qhat\otimes\qhat]`.

        Parameters
        ----------
        state : (r,) ndarray
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        out : (r,) ndarray
            The evaluation :math:`\Ghat[\qhat\otimes\qhat\otimes\qhat]`.
        """
        if self.entries.shape[0] == 1:
            return self.entries[0, 0] * state**3  # r = 1.
        return self.entries @ jnp.prod(state[self._mask], axis=1)

    @utils.requires("entries")
    def jacobian(self, state, input_=None):
        r"""Construct the state Jacobian of the operator:
        :math:`\ddqhat\Ophat_{\ell}(\qhat,\u)
        = \Ghat[(\I_r\otimes\qhat\otimes\qhat)
        + (\qhat\otimes\I_r\otimes\qhat)
        + (\qhat\otimes\qhat\otimes\I_r)]`.

        Parameters
        ----------
        state : (r,) ndarray or None
            State vector.
        input_ : (m,) ndarray or None
            Input vector (not used).

        Returns
        -------
        jac : (r, r) ndarray
            State Jacobian
            :math:`\Ghat[(\I_r\otimes\qhat\otimes\qhat)
            + (\qhat\otimes\I_r\otimes\qhat)
            + (\qhat\otimes\qhat\otimes\I_r)]`.
        """
        if self._prejac is None:
            self._precompute_jacobian_jit()
        q_ = jnp.atleast_1d(state)
        return (self._prejac @ q_) @ q_

    @utils.requires("entries")
    def galerkin(self, Vr, Wr=None):
        r"""Return the Galerkin projection of the operator,
        :math:`\Ghat = (\Wr\trp\Vr)^{-1}\Wr\trp\G[\Vr\otimes\Vr\otimes\Vr]`.

        Parameters
        ----------
        Vr : (n, r) ndarray
            Basis for the trial space.
        Wr : (n, r) ndarray or None
            Basis for the test space. If ``None``, defaults to ``Vr``.

        Returns
        -------
        projected : :class:`opinf.operators.CubicOperator`
            Projected operator.
        """

        def _pg(G, V):
            return self.expand_entries(G) @ jnp.kron(V, jnp.kron(V, V))

        return self._galerkin(Vr, Wr, _pg)

    @staticmethod
    def datablock(states, inputs=None):
        r"""Return the data matrix block corresponding to the operator,
        the Khatri--Rao product of the state with itself three times:
        :math:`\Qhat\odot\Qhat\odot\Qhat` where :math:`\Qhat` is ``states``.

        Since :math:`\Ophat_\ell(\qhat,\u) = \Ohat_{\ell}\d_{\ell}(\qhat,\u)`
        with :math:`\Ohat_{\ell} = \Ghat` and
        :math:`\d_{\ell}(\qhat,\u) = \qhat\otimes\qhat\otimes\qhat`,
        the data block should be

        .. math::
           \D\trp
           = \left[\begin{array}{ccc}
           \d_{\ell}(\qhat_0,\u_0)
           & \cdots &
           \d_{\ell}(\qhat_{k-1},\u_{k-1})
           \end{array}\right]
           = \left[\begin{array}{ccc}
           \qhat_0\otimes\qhat_0\otimes\qhat_0
           & \cdots &
           \qhat_{k-1}\otimes\qhat_{k-1}\otimes\qhat_{k-1}
           \end{array}\right]
           \in \RR^{r^3 \times k}.

        Internally, a compressed triple Kronecker product with
        :math:`r(r+1)(r+2)/6 < r^{3}` degrees of freedom is used for
        efficiency, hence the data block is actually

        .. math::
           \D\trp
           = \left[\begin{array}{ccc}
           \qhat_0\,\hat{\otimes}\,\qhat_0\,\hat{\otimes}\,\qhat_0
           & \cdots &
           \qhat_{k-1}\,\hat{\otimes}\,\qhat_{k-1}\,\hat{\otimes}\,\qhat_{k-1}
           \end{array}\right]
           \in\RR^{r(r+1)(r+2)/6 \times k}.

        Parameters
        ----------
        states : (r, k) or (k,) ndarray
            State vectors. Each column is a single state vector.
            If one dimensional, it is assumed that :math:`r = 1`.
        inputs : (m, k) or (k,) ndarray or None
            Input vectors (not used).

        Returns
        -------
        product_ : (r(r+1)(r+2)/6, k) ndarray
            Compressed triple Khatri--Rao product of ``states`` with itself.
        """
        return CubicOperator.ckron(jnp.atleast_2d(states))

    @staticmethod
    def operator_dimension(r, m=None):
        r"""Column dimension :math:`r(r+1)(r+2)/6` of the internal
        representation :math:`\tilde{\G}` of the operator matrix :math:`\Ghat`.

        Parameters
        ----------
        r : int
            State dimension.
        m : int or None
            Input dimension.
        """
        return r * (r + 1) * (r + 2) // 6

    # Utilities ---------------------------------------------------------------
    @staticmethod
    def ckron(state):
        r"""Calculate the compressed cubic Kronecker product of a vector with
        itself.

        For a vector :math:`\qhat = [~\hat{q}_{1}~~\cdots~~\hat{q}_{r}~]\trp`,
        the cubic Kronecker product of :math:`\qhat` with itself is given by

        .. math::
           \qhat \otimes \qhat \otimes \qhat
           = \left[\begin{array}{c}
               \hat{q}_{1}(\qhat \otimes \qhat)
               \\ \vdots \\
               \hat{q}_{r}(\qhat \otimes \qhat)
           \end{array}\right]
           \in\RR^{r^3}.

        Cross terms :math:`\hat{q}_i \hat{q}_j \hat{q}_k` for :math:`i,j,k`
        not all equal appear multiple times in
        :math:`\qhat\otimes\qhat\otimes\qhat`.
        The *compressed cubic Kronecker product*
        :math:`\qhat\,\hat{\otimes}\,\qhat\,\hat{\otimes}\,\qhat`
        consists of the unique terms of :math:`\qhat\otimes\qhat\otimes\qhat`:

        .. math::
           \qhat\,\hat{\otimes}\,\qhat\,\hat{\otimes}\,\qhat
           = \left[\begin{array}{c}
               \hat{q}_{1}^3
               \\
               \hat{q}_{2}[\![\qhat\,\hat{\otimes}\,\qhat]\!]_{1:2}
               \\ \vdots \\
               \hat{q}_{r}[\![\qhat\,\hat{\otimes}\,\qhat]\!]_{1:r}
           \end{array}\right]
           \in \RR^{r(r+1)(r+2)/6}.

        See :meth:`opinf.operators.QuadraticOperator.ckron`.
        For matrices, the product is computed columnwise.

        Parameters
        ----------
        state : (r,) or (r, k) numpy.ndarray
            State vector or matrix where each column is a state vector.

        Returns
        -------
        product : (r(r+1)(r+2)/6,) or (r(r+1)(r+2)/6, k) ndarray
            The compressed triple Kronecker product of ``state`` with itself.
        """
        state2 = QuadraticOperator.ckron(state)
        lens = special.binom(jnp.arange(2, len(state) + 2), 2).astype(int)
        return jnp.concatenate(
            [state[i] * state2[: lens[i]] for i in range(state.shape[0])],
            axis=0,
        )

    @staticmethod
    def ckron_indices(r):
        """Construct a mask for efficiently computing the compressed Kronecker
        triple product.

        This method provides a faster way to evaluate :meth:`ckron`
        when the state dimension ``r`` is known *a priori*.

        Parameters
        ----------
        r : int
            State dimension.

        Returns
        -------
        mask : ndarray
            Compressed Kronecker product mask.

        Examples
        --------
        >>> from opinf.operators import CubicOperator
        >>> r = 20
        >>> mask = CubicOperator.kron_indices(r)
        >>> q = np.random.random(r)
        >>> np.allclose(CubicOperator.ckron(q), np.prod(q[mask], axis=1))
        True
        """
        mask = jnp.zeros((r * (r + 1) * (r + 2) // 6, 3), dtype=int)
        count = 0
        for i in range(r):
            for j in range(i + 1):
                for k in range(j + 1):
                    mask = mask.at[count, :].set(jnp.array((i, j, k)))
                    count += 1
        return mask

    @staticmethod
    # @jax.jit
    def compress_entries(G):
        r"""Given :math:`\Ghat\in\RR^{a\times r^3}`, construct the matrix
        :math:`\tilde{\G}\in\RR^{a \times r(r+1)(r+2)/6}` such that
        :math:`\Ghat[\qhat\otimes\qhat\otimes\qhat]
        = \tilde{\G}[\qhat\,\hat{\otimes}\,\qhat\,\hat{\otimes}\,\qhat]`
        for all :math:`\qhat\in\RR^{r}`
        where :math:`\cdot\hat{\otimes}\cdot\hat{\otimes}\cdot` is the
        compressed cubic Kronecker product (see :meth:`ckron`).

        Parameters
        ----------
        G : (a, r^3) ndarray
            Matrix that acts on the full cubic Kronecker product.

        Returns
        -------
        Gc : (a, r(r+1)(r+2)/6) ndarray
            Matrix that acts on the compressed cubic Kronecker product.

        Examples
        --------
        >>> from opinf.operators import CubicOperator
        >>> r = 20
        >>> G = np.random.random((r, r**3))
        >>> G.shape
        (20, 8000)
        >>> Gtilde = CubicOperator.compress_entries(G)
        >>> Gtilde.shape
        (20, 1540)
        >>> q = np.random.random(r)
        >>> Gq3 = G @ np.kron(q, np.kron(q, q))
        >>> np.allclose(Gq3, Gtilde @ CubicOperator.ckron(q))
        True
        """
        if jnp.ndim(G) == 1:
            G = jnp.atleast_2d(G)
        r3 = G.shape[1]
        if (r := int(round(r3 ** (1 / 3), 0))) ** 3 != r3:
            raise ValueError(
                f"invalid shape (a, r3) = {G.shape} "
                "with r3 not a perfect cube"
            )
        Gc = jnp.empty((G.shape[0], r * (r + 1) * (r + 2) // 6))

        fj = 0
        for i in range(r):
            for j in range(i + 1):
                for k in range(j + 1):
                    idxs = set(itertools.permutations((i, j, k), 3))
                    Gc = Gc.at[:, fj].set(jnp.sum(
                        jnp.array([G[:, (a * r**2) + (b * r) + c] for a, b, c in idxs]),
                        axis=0,
                    ))
                    fj += 1

        return Gc

    @staticmethod
    # @jax.jit
    def expand_entries(Gc):
        r"""Given :math:`\tilde{\G}\in\RR^{a \times r(r+1)(r+2)/6}`,
        construct the matrix :math:`\Ghat\in\RR^{a\times r^3}` such that
        :math:`\Ghat[\qhat\otimes\qhat\otimes\qhat]
        = \tilde{\G}[\qhat\,\hat{\otimes}\,\qhat\,\hat{\otimes}\,\qhat]`
        for all :math:`\qhat\in\RR^{r}`
        where :math:`\cdot\hat{\otimes}\cdot\hat{\otimes}\cdot` is the
        compressed cubic Kronecker product (see :meth:`ckron`).

        Parameters
        ----------
        Gc : (a, r(r+1)(r+2)/6) ndarray
            Matrix that acts on the compressed cubic Kronecker product.

        Returns
        -------
        G : (a, r^3) ndarray
            Matrix that acts on the full cubic Kronecker product.

        Examples
        --------
        >>> from opinf.operators import CubicOperator
        >>> r = 20
        >>> Gtilde = np.random.random((r, r * (r + 1) * (r + 2)/ 6))
        >>> Gtilde.shape
        (20, 1540)
        >>> G = CubicOperator.expand_entries(Gtilde)
        >>> G.shape
        (20, 8000)
        >>> q = np.random.random(r)
        >>> Gq3 = G @ np.kron(q, np.kron(q, q))
        >>> np.allclose(Gq3, Gtilde @ CubicOperator.ckron(q))
        True
        >>> np.all(CubicOperator.compress_entries(G) == Gtilde)
        True
        """
        if jnp.ndim(Gc) == 1:
            Gc = jnp.atleast_2d(Gc)
        b = Gc.shape[1]
        r = CubicOperator._rfromcompressed(b)
        if r * (r + 1) * (r + 2) // 6 != b:
            raise ValueError(
                f"invalid shape (a, r3) = {Gc.shape} "
                "with r3 != r(r+1)(r+2)/6 for any integer r"
            )

        G = jnp.empty((Gc.shape[0], r**3))
        fj = 0
        for i in range(r):
            for j in range(i + 1):
                for k in range(j + 1):
                    idxs = set(itertools.permutations((i, j, k), 3))
                    fill = Gc[:, fj] / len(idxs)
                    for a, b, c in idxs:
                        G = G.at[:, (a * r**2) + (b * r) + c].set(fill)
                    fj += 1

        return G

    @staticmethod
    def _rfromcompressed(b: int, maxiters: int = 10, tol: float = 0.25) -> int:
        """Compute r such that r(r+1)(r+2)/6 = b via 1D Newton's method."""
        r = int(b ** (1 / 3))
        _6b = 6 * b
        for _ in range(maxiters):
            _3r2 = 3 * r**2
            rnew = r - (r**3 + _3r2 + 2 * r - _6b) / (_3r2 + 6 * r + 2)
            if abs(r - rnew) < tol:
                return int(round(rnew, 0))
            r = rnew
        raise ValueError(  # pragma: no cover
            f"Newton solve for r such that r(r+1)(r+2)/6 = {b} failed"
        )

