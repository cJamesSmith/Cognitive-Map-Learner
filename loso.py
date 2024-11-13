import math
from typing import NamedTuple, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np


class LosseParams(NamedTuple):
    count: jax.Array
    projection: jax.Array
    xtx: jax.Array
    xty: jax.Array
    w: jax.Array


def _to_1d_index(indices, offsets, n_feat, bin_dim, n_bins):
    """Compute the flattened index into the weight matrix."""
    n_grids_per_lsh = (n_bins + 1) ** bin_dim
    indices = jnp.reshape(indices, (-1, bin_dim, n_feat))
    offsets = jnp.reshape(offsets, (-1, bin_dim, n_feat))
    indices = jnp.stack([indices, indices + 1], axis=-1)  # [-1, bin_dim, n_feat, 2]
    values = jnp.stack([1.0 - offsets, offsets], axis=-1)  # [-1, bin_dim, n_feat, 2]
    multiplier = jnp.power(n_bins + 1, jnp.arange(bin_dim - 1, -1, -1))
    indices *= multiplier[:, None, None]
    # shape = (-1, n_feat, ) + (2,) * bin_dim
    shape_suffix = [tuple(*p) for p in np.split(np.eye(bin_dim, dtype=np.int32) + 1, bin_dim)]
    indices = sum(jnp.reshape(indices[:, i], (-1, n_feat, *suffix)) for i, suffix in enumerate(shape_suffix))
    values = math.prod(jnp.reshape(values[:, i], (-1, n_feat, *suffix)) for i, suffix in enumerate(shape_suffix))
    # both indices and values has the shape (-1, n_feat, *(2,)*bin_dim) now.
    indices += jnp.expand_dims(
        n_grids_per_lsh * jnp.arange(n_feat), axis=tuple(range(-bin_dim, 1, 1))
    )  # expand 1 dim in the front and bin_dim in the back.
    indices = jnp.reshape(indices, (-1, n_feat * 2**bin_dim))
    values = jnp.reshape(values, (-1, n_feat * 2**bin_dim))
    return indices, values


class Losse:
    """Linear regressor with LOcality Sensitive Sparse Encoding (Losse).

    We update the linear weights online sparsely following Algorithm.2 in the paper, i.e., computing the incremental closed-form solution based on newly incoming data points.
    """

    def __init__(
        self,
        inout_dims: Tuple[int, int],
        num_features: int,
        num_bins: int,
        bin_dim: int,
        eps: float,
    ) -> None:
        self.num_features = num_features
        self.num_bins = num_bins
        self.bin_dim = bin_dim
        self.inout_dims = inout_dims
        self.eps = eps
        n_edges = num_bins + 1
        n_grids_per_lsh = (n_edges + 1) ** bin_dim
        self.d = n_grids_per_lsh * num_features

    def init(self, rng: jax.random.PRNGKey) -> LosseParams:
        input_dim = self.inout_dims[0]
        output_dim = self.inout_dims[1]
        std = 1 / jnp.sqrt(input_dim)
        projection = std * jax.random.truncated_normal(
            rng,
            -2,
            2,
            (input_dim, self.num_features * self.bin_dim),
        )
        return LosseParams(
            count=jnp.array(0, dtype=jnp.int32),
            projection=projection,
            xtx=jnp.zeros((self.d * self.d,), dtype=projection.dtype),
            xty=jnp.zeros((self.d, output_dim), dtype=projection.dtype),
            w=jnp.zeros((self.d, output_dim), dtype=projection.dtype),
        )

    def update(
        self,
        params: LosseParams,
        x: jax.Array,
        y: jax.Array,
    ) -> LosseParams:
        chex.assert_tree_shape_prefix((x, y), (1,))  # assert non-batched
        indices, values = self._indices_and_values(params.projection, x)
        params = self._update_memory(params, indices, values, y)
        params = self._update_w(params, indices)
        return params

    def predict(self, params: LosseParams, x: jax.Array):
        indices, values = self._indices_and_values(params.projection, x)
        output = params.w[indices] * values[..., None]
        return output.sum(1)

    def _indices_and_values(
        self,
        projection: jax.Array,
        x: jax.Array,
    ):
        h = jnp.matmul(x, projection)
        h = jax.nn.sigmoid(h)
        h = jnp.clip(h, 0, 1) * self.num_bins
        indices = jnp.floor(h).astype(jnp.int32)
        offsets = h - indices
        indices, values = _to_1d_index(
            indices,
            offsets,
            self.num_features,
            self.bin_dim,
            self.num_bins,
        )
        return indices, values

    def _update_memory(
        self,
        params: LosseParams,
        indices: jax.Array,
        values: jax.Array,
        y: jax.Array,
    ) -> LosseParams:
        chex.assert_equal_shape_prefix((indices, values, y), prefix_len=1)
        xtx_indices = (indices * self.d)[:, :, None] + indices[:, None, :]
        xtx_indices = xtx_indices.flatten()
        xty_indices = indices.flatten()
        xtx_updates = values[:, :, None] * values[:, None]
        xtx_updates = xtx_updates.flatten()
        xty_updates = values[:, :, None] * y[:, None, :]
        xty_updates = xty_updates.reshape(-1, y.shape[-1])
        return params._replace(
            xtx=params.xtx.at[xtx_indices].add(xtx_updates),
            xty=params.xty.at[xty_indices].add(xty_updates),
            count=params.count + y.shape[0],
        )

    def _update_w(
        self,
        params: LosseParams,
        indices: jax.Array,
    ) -> LosseParams:
        indices = indices.flatten()
        sub_indices = (indices * self.d)[:, None] + indices[None, :]
        sub_xtx = jnp.reshape(params.xtx[sub_indices], [indices.shape[0]] * 2)
        sub_xty = params.xty[indices]
        a = sub_xtx
        sub_xtxw = jnp.matmul(
            jnp.reshape(params.xtx, (self.d, self.d))[indices],
            params.w,
        )
        b = sub_xty - sub_xtxw + jnp.matmul(sub_xtx, params.w[indices])
        a_norm = a / params.count + self.eps * jnp.eye(len(a))
        b_norm = b / params.count
        solution = jnp.linalg.solve(a_norm, b_norm)
        return params._replace(w=params.w.at[indices].set(solution))


if __name__ == "__main__":
    # Model.
    losse = Losse(
        inout_dims=(1, 1),
        num_features=50,
        num_bins=5,
        bin_dim=2,
        eps=1e-5,
    )

    # losse.init = jax.jit(losse.init)
    # losse.update = jax.jit(losse.update, donate_argnums=(0,))  # donate to avoid copy
    # losse.predict = jax.jit(losse.predict)

    # Data.
    N = 200
    train_n = N // 20
    # training data 20x less coverage than test data
    xs = jax.device_put(jnp.linspace(-jnp.pi, jnp.pi, train_n).reshape(train_n, 1))
    yx = jax.device_put(jnp.sin(xs))
    test_xs = jnp.linspace(-jnp.pi, jnp.pi, N).reshape(N, 1)

    # Init.
    rng = jax.random.PRNGKey(42)
    _rng, rng = jax.random.split(rng)
    model_state = losse.init(_rng)

    # Online update.
    for i in range(train_n):
        x = xs[i : i + 1]
        y = yx[i : i + 1]
        model_state = losse.update(model_state, x, y)

    # Test.
    pred_ys = losse.predict(model_state, test_xs)

    import matplotlib.pyplot as plt

    plt.scatter(test_xs, pred_ys, label="test")
    plt.plot(test_xs, jnp.sin(test_xs), label="ground truth", color="r")
    plt.legend()
    plt.show()