import abc

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
import scipy.sparse as sparse

import opinf


class _TestOperatorTemplate(abc.ABC):
    """Tests for classes that inherit from operators._base.OperatorTemplate."""

    # Setup -------------------------------------------------------------------
    Operator = NotImplemented
    has_inputs = NotImplemented

    @abc.abstractmethod
    def get_operator(self, r: int, m: int = 0):
        """Return a valid operator to test.

        Parameters
        ----------
        r : int > 0
            State dimension.
        m : int or None
            Input dimension. Ignored if the operator does not act on inputs.

        Returns
        -------
        op : Operator
            Instantiated operator.
        """
        raise NotImplementedError

    # Properties --------------------------------------------------------------
    def test_dimensions(self, r=10, m=2):
        """Test state_dimension and input_dimension."""
        if self.has_inputs:
            op = self.get_operator(r, m)
            assert hasattr(op, "input_dimension")
            assert isinstance(op.input_dimension, int)
            assert op.input_dimension == m
        else:
            op = self.get_operator(r)
        assert isinstance(op.state_dimension, int)
        assert op.state_dimension == r

    def test_str(self, r=11, m=3):
        """Lightly test __str__() and _str()."""
        op = self.get_operator(r, m)
        repr(op)
        op._str("q", "u" if self.has_inputs else None)

    # Methods -----------------------------------------------------------------
    # def test_apply_jacobian_galerkin_copy_save_load(self, r=9, m=3):
    #     """Use verify() to test apply(), jacobian(), and galerkin(), copy(),
    #     save(), and load().
    #     """
    #     self.get_operator(r, m).verify(plot=False)

    def _next_key(self):
        """Helper to manage JAX PRNG state internally for tests."""
        if not hasattr(self, "_key"):
            # Initialize the base key on first use
            self._key = jrandom.key(12345)
        # Split the key: save one for future state, return the other
        self._key, subkey = jrandom.split(self._key)
        return subkey


class _TestOpInfOperator(_TestOperatorTemplate):
    """Tests for classes that inherit from operators._base.OpInfOperator."""

    @abc.abstractmethod
    def get_operator(self, r=None, m=None):
        """Return a valid operator to test.

        Parameters
        ----------
        r : int > 0 or None
            State dimension.
            If ``None`` (default), operator entries should not be populated.
            If a positive integer, operator entries should be populated.
        m : int > 0 or None
            Input dimension. Only required if ``r`` is a postive integer
            and the operator acts on inputs.

        Returns
        -------
        op : Operator
            Instantiated operator.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_entries(self, r: int, m=None):
        """Return a valid entries array with the appropriate dimensions.

        Parameters
        ----------
        r : int > 0
            State dimension.
        m : int or None
            Input dimension. Only required if the operator acts on inputs.

        Returns
        -------
        entries : (r, d) ndarray
            Entries array.
        """
        raise NotImplementedError

    def _make_loss(self, op, q):
        if self.has_inputs:
            u = jrandom.uniform(self._next_key(), shape=(op.input_dimension,))
            return lambda entries: jnp.sum(
                self.Operator(entries=entries).apply(q, u) ** 2
            )
        return lambda entries: jnp.sum(self.Operator(entries=entries).apply(q) ** 2)

    def test_grad_apply_wrt_entries(self, r=6, k=10, m=2):
        """Test that jax.grad differentiates correctly through apply()
        with respect to operator entries."""
        op = self.get_operator(r, m) if self.has_inputs else self.get_operator(r)
        q = jrandom.uniform(self._next_key(), shape=(r,))

        loss = self._make_loss(op, q)

        # Analytical gradient via autodiff
        grad_analytical = jax.grad(loss)(op.entries)

        # Numerical gradient via finite differences
        eps = 1e-4
        grad_numerical = jnp.zeros_like(op.entries)
        flat = op.entries.ravel()
        for i in range(flat.size):
            ep = jnp.zeros_like(flat).at[i].set(eps)
            f_plus = loss((flat + ep).reshape(op.entries.shape))
            f_minus = loss((flat - ep).reshape(op.entries.shape))
            grad_numerical = grad_numerical.at[
                jnp.unravel_index(i, op.entries.shape)
            ].set((f_plus - f_minus) / (2 * eps))

        assert jnp.allclose(grad_analytical, grad_numerical, rtol=1e-3, atol=1e-4)
