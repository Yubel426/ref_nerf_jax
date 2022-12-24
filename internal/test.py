import jax
import jax.numpy as jnp


def refractive(viewdirs, normals):
    n_dot_w = jnp.sum(
        normals * viewdirs, axis=-1, keepdims=True)
    index = 1 / 1.5
    return (index * n_dot_w + jnp.sqrt(1 - index ** 2 * (1 - n_dot_w ** 2))) * normals - index * viewdirs


a = jnp.arange(4)
a = jnp.reshape(a, (2,2))
a = a**2
print(a)