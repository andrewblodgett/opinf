import abc
import pytest
import jax
import jax.numpy as jnp
import scipy.sparse as sparse
import jax.random as jrandom

import opinf

import opinf_jax.operators._nonparametric as _module

try:
    from .test_base import _TestOpInfOperator
except ImportError:
    from test_base import _TestOpInfOperator

class _TestNonparametricOperator(_TestOpInfOperator):

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

class TestLinearOperator(_TestNonparametricOperator):
    """Test operators._nonparametric.LinearOperator."""

    Operator = _module.LinearOperator
    has_inputs = False

    def test_set_entries(self):

        # Too many dimensions.
        Abad = jnp.arange(12).reshape((2, 2, 3))
        with pytest.raises(ValueError) as ex:
            op = self.Operator(Abad)
        assert ex.value.args[0] == (
            "LinearOperator entries must be two-dimensional"
        )

        # Nonsquare.
        Abad = Abad.reshape((4, 3))
        with pytest.raises(ValueError) as ex:
            op = self.Operator(Abad)
        assert ex.value.args[0] == (
            "LinearOperator entries must be square (r x r)"
        )

        # Correct square usage.
        A = Abad[:3, :3]
        op = self.Operator(A)
        assert op.entries is A
        assert op.state_dimension == 3

        # Special case: r = 1, scalar A.
        a = jrandom.uniform(self._next_key())
        op = self.Operator(a)
        assert op.shape == (1, 1)
        assert op.state_dimension == 1
        assert op[0, 0] == a

        # Sparse matrix.
        A = jrandom.uniform(self._next_key(), shape=(100, 100))
        A = A.at[A < 0.95].set(0)
        # A = sparse.csr_matrix(A)
        op = self.Operator(A)
        assert op.state_dimension == 100
        assert op.shape == (100, 100)
        # assert sparse.issparse(op.entries)

    def test_apply(self, k=20):
        """Test apply()/__call__()."""

        def _test_single(r):
            A = jrandom.uniform(self._next_key(), shape=(r, r))
            op = self.Operator(A)
            # Evaluation for a single vector.
            q = jrandom.uniform(self._next_key(), shape=r)
            assert jnp.allclose(op.apply(q), A @ q)
            # Vectorized evaluation.
            Q = jrandom.uniform(self._next_key(), shape=(r, k))
            assert jnp.allclose(op.apply(Q), A @ Q)

        _test_single(10)
        _test_single(4)
        _test_single(1)

        # Special case: A is 1x1 and q is a scalar.
        A = jrandom.uniform(self._next_key())
        op = self.Operator(A)
        # Evaluation for a single vector.
        q = jrandom.uniform(self._next_key())
        out = op.apply(q)
        assert jnp.isscalar(out)
        assert jnp.allclose(out, A * q)
        # Vectorized evaluation.
        Q = jrandom.uniform(self._next_key(), shape=k)
        out = op.apply(Q)
        assert out.shape == (k,)
        assert jnp.allclose(out, A * Q)

    def test_jacobian(self, r=9):
        """Test jacobian()."""
        A = jrandom.uniform(self._next_key(), shape=(r, r))
        op = self.Operator(A)
        jac = op.jacobian(jrandom.uniform(self._next_key(), r))
        assert jac.shape == A.shape
        assert jnp.all(jac == A)

    def test_datablock(self, m=3, k=20, r=10):
        """Test datablock()."""
        op = self.get_operator(r)

        state_ = jrandom.uniform(self._next_key(), shape=(r, k))
        input_ = jrandom.uniform(self._next_key(), shape=(m, k))

        assert jnp.array_equal(op.datablock(state_, input_), state_)
        assert jnp.array_equal(op.datablock(state_, None), state_)

        # Special case: r = 1.
        state_ = jrandom.uniform(self._next_key(), shape=k)
        block = op.datablock(state_)
        assert block.shape == (1, k)
        assert jnp.all(block[0] == state_)

    def test_operator_dimension(self):
        """Test operator_dimension()."""
        assert self.Operator.operator_dimension(2) == 2
        assert self.Operator.operator_dimension(4, 6) == 4


class TestQuadraticOperator(_TestNonparametricOperator):
    """Test operators._nonparametric.QuadraticOperator."""
    Operator = _module.QuadraticOperator
    has_inputs = False

    def test_set_entries(self, r=4):
        """Test set_entries()."""
        # op = self.Operator()

        # Too many dimensions.
        Hbad = jnp.arange(16).reshape((2, 2, 2, 2))
        with pytest.raises(ValueError) as ex:
            op = self.Operator(entries=Hbad)
        assert ex.value.args[0] == (
            "QuadraticOperator entries must be two-dimensional"
        )

        # Two-dimensional but invalid shape.
        Hbad = Hbad.reshape((r, r))
        with pytest.raises(ValueError) as ex:
            op = self.Operator(entries=Hbad)
        assert ex.value.args[0] == (
            "invalid QuadraticOperator entries dimensions"
        )

        # Special case: r = 1, H a scalar.
        H = jrandom.uniform(self._next_key())
        op = self.Operator(entries=H)
        assert op.state_dimension == 1
        assert op.shape == (1, 1)
        assert op.entries[0, 0] == H

        # Full operator, compressed internally.
        H = jrandom.uniform(self._next_key(), shape=(r, r**2))
        H_ = self.Operator.compress_entries(H)
        op = self.Operator(entries=H)
        r2_ = r * (r + 1) // 2
        assert op.state_dimension == r
        assert op.shape == (r, r2_)
        assert jnp.allclose(op.entries, H_)

        # Three-dimensional tensor.
        op = self.Operator(entries=H.reshape((r, r, r)))
        assert op.state_dimension == r
        assert op.shape == (r, r2_)
        assert jnp.allclose(op.entries, H_)

        # Compressed operator.
        H = jrandom.uniform(self._next_key(), shape=(r, r2_))
        op = self.Operator(entries=H)
        assert op.entries is H
        assert op.state_dimension == r

        # Test _clear().
        # op._clear()
        # assert op.entries is None
        # assert op._mask is None
        # assert op._prejac is None
        # assert op.state_dimension is None
        # assert op.shape is None

    def test_apply(self, k=10, ntrials=10):
        """Test apply()/__call__()."""
        # op = self.Operator()

        def _test_single(r):
            H = jrandom.uniform(self._next_key(), shape=(r, r**2))
            op = self.Operator(entries=H)
            for _ in range(ntrials):
                # Evaluation for a single vector.
                q = jrandom.uniform(self._next_key(), shape=(r,))
                evaltrue = H @ jnp.kron(q, q)
                evalgot = op.apply(q)
                assert evalgot.shape == (r,)
                assert jnp.allclose(evalgot, evaltrue)
                # Vectorized evaluation.
                Q = jrandom.uniform(self._next_key(), shape=(r, k))
                KR = jnp.column_stack([jnp.kron(Q[:, i], Q[:, i]) for i in range(Q.shape[1])])
                evaltrue = H @ KR
                evalgot = op.apply(Q)
                assert evalgot.shape == (r, k)
                assert jnp.allclose(evalgot, evaltrue)

        _test_single(5)
        _test_single(2)
        _test_single(1)

        # Special case: r = 1 and q is a scalar.
        H = jrandom.uniform(self._next_key())
        op = self.Operator(entries=H)
        for _ in range(ntrials):
            # Evaluation for a single vector.
            q = jrandom.uniform(self._next_key())
            evaltrue = H * q**2
            evalgot = op.apply(q)
            assert jnp.isscalar(evalgot)
            assert jnp.isclose(evalgot, evaltrue)
            # Vectorized evaluation.
            Q = jrandom.uniform(self._next_key(), shape=(k,))
            evaltrue = H * Q**2
            evalgot = op.apply(Q)
            assert evalgot.shape == (k,)
            assert jnp.allclose(evalgot, evaltrue)

    def test_jacobian(self, r=5, ntrials=10):
        """Test jacobian()."""
        H = jrandom.uniform(self._next_key(), shape=(r, r**2))
        op = self.Operator(entries=H)
        # assert op._prejac is None

        # r > 1
        Id = jnp.eye(r)
        for _ in range(ntrials):
            q = jrandom.uniform(self._next_key(), shape=(r,))
            jac_true = H @ (jnp.kron(Id, q) + jnp.kron(q, Id)).T
            jac = op.jacobian(q)
            assert jac.shape == (r, r)
            assert jnp.allclose(jac, jac_true)

        # Special case: r = 1
        H = jrandom.uniform(self._next_key(), shape=(1, 1))
        op = self.Operator(entries=H)
        for _ in range(ntrials):
            q = jrandom.uniform(self._next_key(), shape=(1,))
            jac_true = 2 * H * q
            jac = op.jacobian(q)
            assert jac.shape == (1, 1)
            assert jnp.isclose(jac, jac_true)

    def test_datablock(self, k=20, r=10):
        """Test datablock()."""
        op = self.Operator(self.get_entries(r, r))
        state_ = jrandom.uniform(self._next_key(), shape=(r, k))
        r2_ = r * (r + 1) // 2

        block = op.datablock(state_)
        assert block.shape == (r2_, k)
        entries = jrandom.uniform(self._next_key(), shape=(r, r2_))
        op = self.Operator(entries=entries)
        mult = op.entries @ block
        evald = op.apply(state_)
        assert mult.shape == evald.shape
        assert jnp.allclose(mult, evald)

        # Special case: r = 1.
        state_ = state_[0]
        block = op.datablock(state_)
        assert block.shape == (1, k)
        entries = jrandom.uniform(self._next_key())
        op = self.Operator(entries=entries)
        mult = op.entries[0, 0] * block[0]
        evald = op.apply(state_)
        assert mult.shape == evald.shape
        assert jnp.allclose(mult, evald)

    def test_operator_dimension(self):
        """Test operator_dimension()."""
        assert self.Operator.operator_dimension(1) == 1
        assert self.Operator.operator_dimension(3) == 6
        assert self.Operator.operator_dimension(5, 7) == 15

    def test_ckron(self, n_tests=20):
        """Test ckron()."""

        def _check(q, q2):
            for i in range(len(q)):
                assert jnp.allclose(
                    q2[i * (i + 1) // 2 : (i + 1) * (i + 2) // 2],
                    q[i] * q[: i + 1],
                )

        for _ in range(n_tests):
            r = int(jrandom.randint(self._next_key(), (), 2, 10))
            q = jrandom.uniform(self._next_key(), shape=(r,))
            q2 = self.Operator.ckron(q)
            r2 = r * (r + 1) // 2
            assert q2.shape == (r2,)
            _check(q, q2)

            k = int(jrandom.randint(self._next_key(), (), 1, 10))
            Q = jrandom.uniform(self._next_key(), shape=(r, k))
            Q2 = self.Operator.ckron(Q)
            assert Q2.shape == (r2, k)
            _check(Q, Q2)

    def test_ckron_indices(self, n_tests=20):
        """Test ckron_indices()."""
        # Manufactured test.
        mask = self.Operator.ckron_indices(4)
        assert jnp.all(
            mask
            == jnp.array(
                [
                    [0, 0],
                    [1, 0],
                    [1, 1],
                    [2, 0],
                    [2, 1],
                    [2, 2],
                    [3, 0],
                    [3, 1],
                    [3, 2],
                    [3, 3],
                ],
                dtype=int,
            )
        )
        submask = self.Operator.ckron_indices(3)
        assert jnp.allclose(submask, mask[: submask.shape[0]])

        # Random test.
        for _ in range(n_tests):
            r = int(jrandom.randint(self._next_key(), (), 2, 10))
            _r2 = r * (r + 1) // 2
            mask = self.Operator.ckron_indices(r)
            assert mask.shape == (_r2, 2)
            assert mask.sum(axis=0)[0] == sum(i * (i + 1) for i in range(r))
            q = jrandom.uniform(self._next_key(), shape=(r,))
            assert jnp.allclose(
                jnp.prod(q[mask], axis=1), self.Operator.ckron(q)
            )

    def test_compress_entries(self, n_tests=20):
        """Test compress_entries()."""
        # Try with bad second dimension.
        r = 5
        r2bad = r**2 + 1
        H = jnp.empty((r, r2bad))
        with pytest.raises(ValueError) as ex:
            self.Operator.compress_entries(H)
        assert ex.value.args[0] == (
            f"invalid shape (a, r2) = {(r, r2bad)} "
            "with r2 not a perfect square"
        )

        # One-dimensional H (r = 1).
        Hc = self.Operator.compress_entries(jnp.asarray([5]))
        assert Hc.shape == (1, 1)
        assert Hc[0, 0] == 5

        # Random tests.
        for _ in range(n_tests):
            r = int(jrandom.randint(self._next_key(), (), 2, 10))
            # Check dimensions.
            a = int(jrandom.randint(self._next_key(), (), 2, 10))
            H = jrandom.uniform(self._next_key(), shape=(a, r**2))
            r2 = r * (r + 1) // 2
            Hc = self.Operator.compress_entries(H)
            assert Hc.shape == (a, r2)

            # Check that Hc(q^2) == H(q ⊗ q).
            for _ in range(5):
                q = jrandom.uniform(self._next_key(), shape=(r,))
                Hq2 = H @ jnp.kron(q, q)
                assert jnp.allclose(Hq2, Hc @ self.Operator.ckron(q))

            # Check that expand_entries() and compress_quadrati()
            # are inverses up to symmetry.
            H2 = self.Operator.expand_entries(Hc)
            Ht = jnp.reshape(H, (a, r, r))
            H2sym = jnp.reshape(jnp.stack([(Ht[i] + Ht[i].T) / 2 for i in range(a)]), H.shape)
            assert jnp.allclose(H2, H2sym)

    def test_expand_entries(self, n_tests=20):
        """Test expand_entries()."""
        # Try with bad second dimension.
        r = 5
        r2bad = (r * (r + 1) // 2) + 1
        Hc = jnp.empty((r, r2bad))
        with pytest.raises(ValueError) as ex:
            self.Operator.expand_entries(Hc)
        assert ex.value.args[0] == (
            f"invalid shape (a, r2) = {(r, r2bad)} "
            "with r2 != r(r+1)/2 for any integer r"
        )

        # One-dimensional H (r = 1).
        H = self.Operator.expand_entries(jnp.asarray([5]))
        assert H.shape == (1, 1)
        assert H[0, 0] == 5

        # Random tests.
        for _ in range(n_tests):
            r = int(jrandom.randint(self._next_key(), (), 2, 10))
            # Check dimensions.
            a = int(jrandom.randint(self._next_key(), (), 2, 10))
            Hc = jrandom.uniform(self._next_key(), shape=(a, r * (r + 1) // 2))
            H = self.Operator.expand_entries(Hc)
            assert H.shape == (a, r**2)

            # Check that Hc(q^2) == H(q ⊗ q).
            for _ in range(5):
                q = jrandom.uniform(self._next_key(), shape=(r,))
                Hq2 = H @ jnp.kron(q, q)
                assert jnp.allclose(Hq2, Hc @ self.Operator.ckron(q))

            # Check that expand_entries() and compress_entries() are inverses.
            Hc2 = self.Operator.compress_entries(H)
            assert jnp.allclose(Hc2, Hc)

class TestCubicOperator(_TestNonparametricOperator):
    """Test operators._nonparametric.CubicOperator."""

    Operator = _module.CubicOperator
    has_inputs = False

    def test_set_entries(self, r=4):
        """Test set_entries()."""

        # Too many dimensions.
        Gbad = jnp.arange(4).reshape((1, 2, 1, 2))
        with pytest.raises(ValueError) as ex:
            op = self.Operator(Gbad)
        assert ex.value.args[0] == (
            "CubicOperator entries must be two-dimensional"
        )

        # Two-dimensional but invalid shape.
        Gbad = jrandom.uniform(self._next_key(), (3, 8))
        with pytest.raises(ValueError) as ex:
            op = self.Operator(Gbad)
        assert ex.value.args[0] == "invalid CubicOperator entries dimensions"

        # Special case: r = 1
        G = jrandom.uniform(self._next_key(), (1, 1))
        op = self.Operator(G)
        assert op.shape == (1, 1)
        assert jnp.allclose(op.entries, G)

        # Full operator, compressed internally.
        G = jrandom.uniform(self._next_key(), (r, r**3))
        G_ = self.Operator.compress_entries(G)
        op = self.Operator(G)
        r3_ = r * (r + 1) * (r + 2) // 6
        assert op.shape == (r, r3_)
        assert jnp.allclose(op.entries, G_)

        # Three-dimensional tensor.
        op = self.Operator(G.reshape((r, r, r, r)))
        assert op.shape == (r, r3_)
        assert jnp.allclose(op.entries, G_)

        # Compressed operator.
        G = jrandom.uniform(self._next_key(), (r, r3_))
        op = self.Operator(G)
        assert op.entries is G

        # Test _clear().
        # op._clear()
        # assert op.entries is None
        # assert op._mask is None
        # assert op._prejac is None


    def test_apply(self, k=20, ntrials=10):
        """Test apply()/__call__()."""
        def khatri_rao(A, B):
            compute_cols = jax.vmap(lambda a, b: jnp.outer(b, a).ravel(), in_axes=1, out_axes=1)
            return compute_cols(A, B)

        def _test_single(r):
            G = jrandom.uniform(self._next_key(), (r, r**3))
            op = self.Operator(G)
            for _ in range(ntrials):
                # Evaluation for a single vector.
                q = jrandom.uniform(self._next_key(), r)
                evaltrue = G @ jnp.kron(jnp.kron(q, q), q)
                evalgot = op.apply(q)
                assert jnp.allclose(evalgot, evaltrue)
                # Vectorized evaluation.
                Q = jrandom.uniform(self._next_key(), (r, k))
                evaltrue = G @ khatri_rao(Q, khatri_rao(Q, Q))
                evalgot = op.apply(Q)
                assert evalgot.shape == (r, k)
                assert jnp.allclose(evalgot, evaltrue)

        _test_single(5)
        _test_single(3)
        _test_single(1)

        # Special case: r = 1 and q is a scalar.
        G = jrandom.uniform(self._next_key(), )
        op = self.Operator(G)
        for _ in range(ntrials):
            # Evaluation for a single vector.
            q = jrandom.uniform(self._next_key(), )
            evaltrue = G * q**3
            evalgot = op.apply(q)
            assert jnp.isscalar(evalgot)
            assert jnp.allclose(evalgot, evaltrue)
            # Vectorized evaluation.
            Q = jrandom.uniform(self._next_key(), k)
            evaltrue = G * Q**3
            evalgot = op.apply(Q)
            assert evalgot.shape == (k,)
            assert jnp.allclose(evalgot, evaltrue)

    def test_jacobian(self, r=5, ntrials=10):
        """Test jacobian()."""
        G = jrandom.uniform(self._next_key(), (r, r**3))
        op = self.Operator(G)
        assert op._prejac is None

        Id = jnp.eye(r)
        for _ in range(ntrials):
            q = jrandom.uniform(self._next_key(), r)
            qId = jnp.kron(q, Id)
            Idq = jnp.kron(Id, q)
            qqId = jnp.kron(q, qId)
            qIdq = jnp.kron(qId, q)
            Idqq = jnp.kron(Idq, q)
            jac_true = G @ (Idqq + qIdq + qqId).T
            jac = op.jacobian(q)
            assert jac.shape == (r, r)
            assert jnp.allclose(jac, jac_true)

        # Special case: r = 1
        G = jrandom.uniform(self._next_key(), )
        op = self.Operator(G)
        for _ in range(ntrials):
            q = jrandom.uniform(self._next_key(), )
            jac_true = 3 * G * q**2
            jac = op.jacobian(q)
            assert jac.shape == (1, 1)
            assert jnp.isclose(jac[0, 0], jac_true)

    def test_datablock(self, k=20, r=10):
        """Test datablock()."""
        op = self.Operator()
        state_ = jrandom.uniform(self._next_key(), (r, k))
        r3_ = r * (r + 1) * (r + 2) // 6

        # More thorough tests elsewhere for ckron().
        block = op.datablock(state_)
        assert block.shape == (r3_, k)
        op.entries = jrandom.uniform(self._next_key(), (r, r3_))
        mult = op.entries @ block
        evald = op.apply(state_)
        assert mult.shape == evald.shape
        assert jnp.allclose(mult, evald)

        # Special case: r = 1.
        state_ = state_[0]
        block = op.datablock(state_)
        assert block.shape == (1, k)
        op.entries = jrandom.uniform(self._next_key(), )
        mult = op.entries[0, 0] * block[0]
        evald = op.apply(state_)
        assert mult.shape == evald.shape
        assert jnp.allclose(mult, evald)

    def test_operator_dimension(self):
        """Test operator_dimension()."""
        assert self.Operator.operator_dimension(1) == 1
        assert self.Operator.operator_dimension(3) == 10
        assert self.Operator.operator_dimension(5, 2) == 35

    def test_ckron(self, n_tests=20):
        """Test ckron()."""

        def _check(q, q3):
            for i in range(len(q)):
                assert jnp.allclose(
                    q3[
                        i
                        * (i + 1)
                        * (i + 2)
                        // 6 : (i + 1)
                        * (i + 2)
                        * (i + 3)
                        // 6
                    ],
                    q[i] * TestQuadraticOperator.Operator.ckron(q[: i + 1]),
                )

        for r in jrandom.randint(self.next_key(), 2, 10, n_tests):
            q = jrandom.uniform(self._next_key(), r)
            q3 = self.Operator.ckron(q)
            r3 = r * (r + 1) * (r + 2) // 6
            assert q3.shape == (r3,)
            _check(q, q3)

            k = jrandom.randint(self._next_key(), 1, 10)
            Q = jrandom.uniform(self._next_key(), (r, k))
            Q3 = self.Operator.ckron(Q)
            assert Q3.shape == (r3, k)
            _check(Q, Q3)

    def test_ckron_indices(self, n_tests=20):
        """Test ckron_indices()."""
        # Manufactured test.
        mask = self.Operator.ckron_indices(2)
        assert jnp.all(
            mask
            == jnp.array(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
                dtype=int,
            )
        )

        # Random tests.
        for _ in range(n_tests):
            r = jrandom.randint(self._next_key(), 2, 10)
            mask = self.Operator.ckron_indices(r)
            _r3 = r * (r + 1) * (r + 2) // 6
            mask = self.Operator.ckron_indices(r)
            assert mask.shape == (_r3, 3)
            q = jrandom.uniform(self._next_key(), r)
            assert jnp.allclose(
                jnp.prod(q[mask], axis=1), self.Operator.ckron(q)
            )

    def test_compress_entries(self, n_tests=20):
        """Test compress_entries()."""
        # Try with bad second dimension.
        r = 5
        r3bad = r**3 + 1
        G = jnp.empty((r, r3bad))
        with pytest.raises(ValueError) as ex:
            self.Operator.compress_entries(G)
        assert ex.value.args[0] == (
            f"invalid shape (a, r3) = {(r, r3bad)} "
            "with r3 not a perfect cube"
        )

        # One-dimensional G (r = 1).
        Gc = self.Operator.compress_entries([6])
        assert Gc.shape == (1, 1)
        assert Gc[0, 0] == 6

        # Random tests.
        for r in jrandom.randint(self._next_key(), 2, 10, n_tests):
            # Check dimensions.
            a = jrandom.randint(self._next_key(), 2, 10)
            G = jrandom.uniform(self._next_key(), (a, r**3))
            r2 = r * (r + 1) * (r + 2) // 6
            Gc = self.Operator.compress_entries(G)
            assert Gc.shape == (a, r2)

            # Check that Gc(q^3) == G(q ⊗ q ⊗ q).
            for _ in range(5):
                q = jrandom.uniform(self._next_key(), r)
                Gq3 = G @ jnp.kron(q, jnp.kron(q, q))
                assert jnp.allclose(Gq3, Gc @ self.Operator.ckron(q))

    def test_expand_entries(self, n_tests=20):
        """Test expand_entries()."""
        # Try with bad second dimension.
        r = 5
        r3bad = (r * (r + 1) * (r + 2) // 6) + 1
        Gc = jnp.empty((r, r3bad))
        with pytest.raises(ValueError) as ex:
            self.Operator.expand_entries(Gc)
        assert ex.value.args[0] == (
            f"invalid shape (a, r3) = {(r, r3bad)} "
            "with r3 != r(r+1)(r+2)/6 for any integer r"
        )

        # One-dimensional G (r = 1).
        G = self.Operator.expand_entries([5])
        assert G.shape == (1, 1)
        assert G[0, 0] == 5

        # Random tests.
        for r in jrandom.randint(self._next_key(), 2, 10, n_tests):
            # Check dimensions.
            a = jrandom.randint(self._next_key(), 2, 10)
            Gc = jrandom.uniform(self._next_key(), (a, r * (r + 1) * (r + 2) // 6))
            G = self.Operator.expand_entries(Gc)
            assert G.shape == (a, r**3)

            # Check that Gc[q^3] == G[q ⊗ q ⊗ q].
            for _ in range(5):
                q = jrandom.uniform(self._next_key(), r)
                Gq3 = G @ jnp.kron(q, jnp.kron(q, q))
                assert jnp.allclose(Gq3, Gc @ self.Operator.ckron(q))

            # Check that expand_entries() and compress_entries() are inverses.
            Gc2 = self.Operator.compress_entries(G)
            assert jnp.allclose(Gc2, Gc)
















