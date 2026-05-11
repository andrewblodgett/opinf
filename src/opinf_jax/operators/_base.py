import abc
import types
import jax
import jax.numpy as jnp
import equinox as eqx

from .. import errors, utils

class OperatorTemplate(eqx.Module):
    @abc.abstractmethod
    def apply(self, state: jax.Array, input_=None) -> jax.Array:
        raise NotImplementedError
    
class OpInfOperator(OperatorTemplate):
    _entries: jax.Array = eqx.field(converter=jnp.asarray)

    def __init__(self, entries):
        OpInfOperator._validate_entries(entries)
        self._entries = entries

    @staticmethod
    def _validate_entries(entries):
        """Ensure argument is a jax array and screen for NaN, Inf entries."""
        if not isinstance(entries, jax.Array):
            raise TypeError(
                "operator entries must be jax.Array"
            )
        if jnp.any(jnp.isnan(entries)):
            raise ValueError("operator entries must not be NaN")
        elif jnp.any(jnp.isinf(entries)):
            raise ValueError("operator entries must not be Inf")

    @property
    def entries(self) -> jax.Array:
        return self._entries
    
    @property
    def shape(self) -> tuple:
        """Shape of the operator matrix."""
        return None if self.entries is None else self.entries.shape

    @property
    def state_dimension(self) -> int:
        r"""Dimension :math:`r` of the state :math:`\qhat` that the operator
        acts on.
        """
        return None if self.entries is None else self.entries.shape[0]

    @utils.requires("entries")
    def jacobian(self, state, input_=None) -> jax.Array:  # pragma: no cover
        return 0
    
    def _galerkin(self, Vr, Wr, func):
        if Wr is None:
            Wr = Vr
        n, r = Wr.shape
        if self.entries.shape[0] != n:  # pragma: no cover
            raise errors.DimensionalityError("basis and operator not aligned")
        if Vr.shape[1] != r:  # pragma: no cover
            raise errors.DimensionalityError(
                "trial and test bases not aligned"
            )

        entries = Wr.T @ func(self.entries, Vr)
        if not jnp.allclose((WrTVr := Wr.T @ Vr), jnp.eye(r)):
            entries = jnp.linalg.solve(WrTVr, entries)
        return self.__class__(entries)