import abc
import pytest
import jax.numpy as jnp

import jax.random as jrandom

import opinf

import opinf_jax.operators._nonparametric as _module

try:
    from .test_base import _TestOpInfOperator
except ImportError:
    from test_base import _TestOpInfOperator

class _TestNonparametricOperator(_TestOpInfOperator):
    
    def _next_key(self):
        """Helper to manage JAX PRNG state internally for tests."""
        if not hasattr(self, '_key'):
            # Initialize the base key on first use
            self._key = jrandom.key(12345)
        # Split the key: save one for future state, return the other
        self._key, subkey = jrandom.split(self._key)
        return subkey

    def get_operator(self, r, m=None):
        return self.Operator(entries=self.get_entries(r, m))

    def get_entries(self, r, m):
        d = self.Operator.operator_dimension(r, m)
        return jrandom.normal(self._next_key(), shape=(r, d))
    
    @abc.abstractmethod
    def test_set_entries(self):
        raise NotImplementedError

    def test_in_model(self, r=3, k=100, m=2):
        """See if we can fit a model with this operator."""
        model = opinf.models.ContinuousModel(operators=[self.get_operator(r)])
        
        Q = jrandom.uniform(self._next_key(), shape=(r, k))
        dQ = jrandom.uniform(self._next_key(), shape=(r, k))

        if self.has_inputs:
            U = jrandom.uniform(self._next_key(), shape=(m, k))
            model = model.fit(states=Q, ddts=dQ, inputs=U)
        else:
            model = model.fit(states=Q, ddts=dQ, inputs=None)


class TestConstantOperator(_TestNonparametricOperator):
    """Test operators._nonparametric.ConstantOperator."""

    Operator = _module.ConstantOperator
    has_inputs = False

    def test_set_entries(self):
        """Test set_entries()."""
        # op = self.Operator()

        # Too many dimensions.
        cbad = jnp.arange(12).reshape((4, 3))
        with pytest.raises(ValueError) as ex:
            # op.set_entries(cbad)
            op = self.Operator(entries=cbad)
        assert ex.value.args[0] == (
            "ConstantOperator entries must be one-dimensional"
        )

        # Case 1: one-dimensional array.
        c = jnp.arange(12)
        op = self.Operator(entries=c)
        assert op.entries is c

        # Case 2: two-dimensional array that can be flattened.
        op = self.Operator(entries=c.reshape((-1,1)))
        # op.set_entries(c.reshape((-1, 1)))
        assert op.shape == (12,)
        assert op.state_dimension == 12
        assert jnp.all(op.entries == c)

        # op.set_entries(c.reshape((1, -1)))
        op = self.Operator(entries=c.reshape((1,-1)))
        assert op.shape == (12,)
        assert op.state_dimension == 12
        assert jnp.all(op.entries == c)

        # Case 3: r = 1 and c is a scalar.
        c = jrandom.uniform(self._next_key()) 
        # op.set_entries(c)
        op = self.Operator(entries=c)
        assert op.shape == (1,)
        assert op.state_dimension == 1
        assert op.entries[0] == c

    def test_apply(self, k=20):
        """Test apply()/__call__()."""
        # op = self.Operator()
        
        def _test_single(r):
            c = jrandom.uniform(self._next_key(), shape=(r,))
            op = self.Operator(entries=c)
            assert jnp.allclose(op.apply(), c)
            
            # Evaluation for a single vector.
            q = jrandom.uniform(self._next_key(), shape=(r,))
            assert jnp.allclose(op.apply(q), op.entries)
            
            # Vectorized evaluation.
            Q = jrandom.uniform(self._next_key(), shape=(r, k))
            ccc = jnp.column_stack([c for _ in range(k)])
            out = op.apply(Q)
            assert out.shape == (r, k)
            assert jnp.all(out == ccc)

        _test_single(10)
        _test_single(20)
        _test_single(1)

        # Special case: r = 1, scalar q.
        c = jrandom.uniform(self._next_key(), shape=())
        # op.set_entries(c)
        op = self.Operator(entries=c)
        
        # Evaluation for a single vector.
        q = jrandom.uniform(self._next_key(), shape=())
        out = op.apply(q)
        assert jnp.isscalar(out)
        assert out == c
        
        # Vectorized evaluation.
        Q = jrandom.uniform(self._next_key(), shape=(k,))
        out = op.apply(Q)
        assert out.shape == (k,)
        assert jnp.all(out == c)

    def test_datablock(self, k=20):
        """Test datablock()."""
        op = self.Operator(1)
        ones = jnp.ones((1, k))
        
        def _test_single(out):
            assert out.shape == ones.shape
            assert jnp.all(out == ones)

        _test_single(op.datablock(jrandom.uniform(self._next_key(), shape=(5, k))))
        _test_single(op.datablock(jrandom.uniform(self._next_key(), shape=(3, k))))
        _test_single(op.datablock(jrandom.uniform(self._next_key(), shape=(k,))))

    def test_operator_dimension(self):
        """Test operator_dimension()."""
        assert self.Operator.operator_dimension() == 1
        assert self.Operator.operator_dimension(4) == 1
        assert self.Operator.operator_dimension(1, 6) == 1