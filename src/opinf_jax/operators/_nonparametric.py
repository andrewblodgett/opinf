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