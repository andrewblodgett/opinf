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



