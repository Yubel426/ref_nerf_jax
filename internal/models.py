from flax import linen as nn
from jax import random
import jax
import jax.numpy as jnp
import functools
from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Text, Tuple
import utils
import image
import gin
import mip


def random_split(rng):
    if rng is None:
        key = None
    else:
        key, rng = random.split(rng)
    return key, rng


class MipNerfModel(nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""
    num_samples: int = 128  # The number of samples per level.
    num_levels: int = 2  # The number of sampling levels.
    resample_padding: float = 0.01  # Dirichlet/alpha "padding" on the histogram.
    stop_level_grad: bool = True  # If True, don't backprop across levels')
    use_viewdirs: bool = True  # If True, use view directions as a condition.
    lindisp: bool = False  # If True, sample linearly in disparity, not in depth.
    ray_shape: str = 'cone'  # The shape of cast rays ('cone' or 'cylinder').
    min_deg_point: int = 0  # Min degree of positional encoding for 3D points.
    max_deg_point: int = 16  # Max degree of positional encoding for 3D points.
    deg_view: int = 4  # Degree of positional encoding for viewdirs.
    density_activation: Callable[..., Any] = nn.softplus  # Density activation.
    density_noise: float = 0.  # Standard deviation of noise added to raw density.
    density_bias: float = -1.  # The shift added to raw densities pre-activation.
    rgb_activation: Callable[..., Any] = nn.sigmoid  # The RGB activation.
    rgb_padding: float = 0.001  # Padding added to the RGB outputs.
    disable_integration: bool = False  # If True, use PE instead of IPE.

    @nn.compact
    def __call__(self, rng, rays, randomized, white_bkgd):
        """The mip-NeRF Model.

    Args:
      rng: jnp.ndarray, random number generator.
      rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
      randomized: bool, use randomized stratified sampling.
      white_bkgd: bool, if True, use white as the background (black o.w.).

    Returns:
      ret: list, [*(rgb, distance, acc)]
    """
        # Construct the MLP.
        mlp = MLP()
        ray_history = []
        ret = []
        for i_level in range(self.num_levels):
            key, rng = random.split(rng)
            if i_level == 0:
                # Stratified sampling along rays
                t_vals, samples = mip.sample_along_rays(
                    key,
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    self.num_samples,
                    rays.near,
                    rays.far,
                    randomized,
                    self.lindisp,
                    self.ray_shape,
                )
            else:
                t_vals, samples = mip.resample_along_rays(
                    key,
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    t_vals,
                    weights,
                    randomized,
                    self.ray_shape,
                    self.stop_level_grad,
                    resample_padding=self.resample_padding,
                )
            if self.disable_integration:
                samples = (samples[0], jnp.zeros_like(samples[1]))
            samples_enc = mip.integrated_pos_enc(
                samples,
                self.min_deg_point,
                self.max_deg_point,
            )

            # Point attribute predictions

            ray_results = mlp(
                key,
                samples,
                viewdirs=rays.viewdirs if self.use_viewdirs else None)

            comp_rgb, distance, acc, weights = mip.volumetric_rendering(
                ray_results['rgb'],
                ray_results['density'],
                t_vals,
                rays.directions,
                white_bkgd=white_bkgd,
            )
            ret.append((comp_rgb, distance, acc))
            ray_history.append(ray_results)

        return ret, ray_history


def construct_mipnerf(rng, example_batch):
    """Construct a Neural Radiance Field.

  Args:
    rng: jnp.ndarray. Random number generator.
    example_batch: dict, an example of a batch of data.

  Returns:
    model: nn.Model. Nerf model with parameters.
    state: flax.Module.state. Nerf model state for stateful parameters.
  """
    model = MipNerfModel()
    key, rng = random.split(rng)
    init_variables = model.init(
        key,
        rng=rng,
        rays=utils.namedtuple_map(lambda x: x[0], example_batch['rays']),
        randomized=False,
        white_bkgd=False)
    return model, init_variables


@gin.configurable
class MLP(nn.Module):
    """A PosEnc MLP."""
    net_depth: int = 8  # The depth of the first part of MLP.
    net_width: int = 256  # The width of the first part of MLP.
    bottleneck_width: int = 256  # The width of the bottleneck vector.
    net_depth_condition: int = 1  # The depth of the second part of MLP.
    net_width_condition: int = 128  # The width of the second part of MLP.
    net_activation: Callable[..., Any] = nn.relu  # The activation function.
    skip_layer: int = 4  # Add a skip connection to the output of every N layers.
    num_rgb_channels: int = 3  # The number of RGB channels.
    num_density_channels: int = 1  # The number of density channels.
    bottleneck_noise: float = 0.0  # Std. deviation of noise added to bottleneck.
    density_activation: Callable[..., Any] = nn.softplus  # Density activation.
    density_bias: float = -1.  # Shift added to raw densities pre-activation.
    density_noise: float = 0.  # Standard deviation of noise added to raw density.
    rgb_premultiplier: float = 1.  # Premultiplier on RGB before activation.
    rgb_activation: Callable[..., Any] = nn.sigmoid  # The RGB activation.
    rgb_bias: float = 0.  # The shift added to raw colors pre-activation.
    rgb_padding: float = 0.001  # Padding added to the RGB outputs.
    disable_rgb: bool = False  # If True don't output RGB.
    weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.
    use_n_dot_v: bool = False  # If True, feed dot(n * viewdir) to 2nd MLP.

    @nn.compact
    def __call__(self,
                 rng,
                 gaussians,
                 viewdirs=None):
        """Evaluate the MLP.

    Args:
      x: jnp.ndarray(float32), [batch, num_samples, feature], points.
      condition: jnp.ndarray(float32), [batch, feature], if not None, this
        variable will be part of the input to the second part of the MLP
        concatenated with the output vector of the first part of the MLP. If
        None, only the first part of the MLP will be used with input x. In the
        original paper, this variable is the view direction.

    Returns:
      raw_rgb: jnp.ndarray(float32), with a shape of
           [batch, num_samples, num_rgb_channels].
      raw_density: jnp.ndarray(float32), with a shape of
           [batch, num_samples, num_density_channels].
    """
        dense_layer = functools.partial(
            nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)())

        density_key, rng = random_split(rng)

        def predict_density(means, covs):
            """Helper function to output density."""
            # Encode input positions

            x = mip.integrated_pos_enc(means, covs,
                                       self.min_deg_point, self.max_deg_point)
            inputs = x

            # Evaluate network to produce the output density.
            for i in range(self.net_depth):
                x = dense_layer(self.net_width)(x)
                x = self.net_activation(x)
                if i % self.skip_layer == 0 and i > 0:
                    x = jnp.concatenate([x, inputs], axis=-1)
            raw_density = dense_layer(1)(x)[..., 0]  # Hardcoded to a single channel.
            # Add noise to regularize the density predictions if needed.
            if (density_key is not None) and (self.density_noise > 0):
                raw_density += self.density_noise * random.normal(
                    density_key, raw_density.shape)
            return raw_density, x

        means, covs = gaussians

        # Flatten the input so value_and_grad can be vmap'ed.
        means_flat = means.reshape((-1, means.shape[-1]))
        covs_flat = covs.reshape((-1,) + covs.shape[len(means.shape) - 1:])

        # Evaluate the network and its gradient on the flattened input.
        predict_density_and_grad_fn = jax.vmap(
            jax.value_and_grad(predict_density, has_aux=True), in_axes=(0, 0))
        (raw_density_flat, x_flat), raw_grad_density_flat = (
            predict_density_and_grad_fn(means_flat, covs_flat))

        # Unflatten the output.
        raw_density = raw_density_flat.reshape(means.shape[:-1])
        x = x_flat.reshape(means.shape[:-1] + (x_flat.shape[-1],))
        raw_grad_density = raw_grad_density_flat.reshape(means.shape)

        # Compute normal vectors as negative normalized density gradient.
        # We normalize the gradient of raw (pre-activation) density because
        # it's the same as post-activation density, but is more numerically stable
        # when the activation function has a steep or flat gradient.
        normals = -utils.l2_normalize(raw_grad_density)

        grad_pred = dense_layer(3)(x)

        # Normalize negative predicted gradients to get predicted normal vectors.
        normals_pred = -utils.l2_normalize(grad_pred)
        normals_to_use = normals_pred

        # Apply bias and activation to raw density
        density = self.density_activation(raw_density + self.density_bias)

        if viewdirs is not None:
            # Predict diffuse color.

            raw_rgb_diffuse = dense_layer(self.num_rgb_channels)(x)

            tint = nn.sigmoid(dense_layer(3)(x))

            # Output of the first part of MLP.
            if self.bottleneck_width > 0:
                bottleneck = dense_layer(self.bottleneck_width)(x)

                # Add bottleneck noise.
                if (rng is not None) and (self.bottleneck_noise > 0):
                    key, rng = random_split(rng)
                    bottleneck += self.bottleneck_noise * random.normal(
                        key, bottleneck.shape)

                x = [bottleneck]
            else:
                x = []

            # Encode view (or reflection) directions.
            # Compute reflection directions. Note that we flip viewdirs before
            # reflecting, because they point from the camera to the point,
            # whereas ref_utils.reflect() assumes they point toward the camera.
            # Returned refdirs then point from the point to the environment.
            refdirs = utils.refractive(-viewdirs[..., None, :], normals_to_use)
            # Encode reflection directions.
            dir_enc = mip.pos_enc(refdirs, 0, self.deg_view)

            # Append view (or reflection) direction encoding to bottleneck vector.
            x.append(dir_enc)

            # Append dot product between normal vectors and view directions.
            if self.use_n_dot_v:
                dotprod = jnp.sum(
                    normals_to_use * viewdirs[..., None, :], axis=-1, keepdims=True)
                x.append(dotprod)

            # Concatenate bottleneck, directional encoding, and GLO.
            x = jnp.concatenate(x, axis=-1)

            # Output of the second part of MLP.
            inputs = x
            for i in range(self.net_depth_viewdirs):
                x = dense_layer(self.net_width_viewdirs)(x)
                x = self.net_activation(x)
                if i % self.skip_layer_dir == 0 and i > 0:
                    x = jnp.concatenate([x, inputs], axis=-1)

        # If using diffuse/specular colors, then `rgb` is treated as linear
        # specular color. Otherwise it's treated as the color itself.
        rgb = self.rgb_activation(self.rgb_premultiplier *
                                  dense_layer(self.num_rgb_channels)(x) +
                                  self.rgb_bias)

        # Initialize linear diffuse color around 0.25, so that the combined
        # linear color is initialized around 0.5.
        diffuse_linear = nn.sigmoid(raw_rgb_diffuse - jnp.log(3.0))
        specular_linear = tint * rgb

        # Combine specular and diffuse components and tone map to sRGB.
        rgb = jnp.clip(
            image.linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0)

        # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
        rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

        return dict(
            density=density,
            rgb=rgb,
            raw_grad_density=raw_grad_density,
            grad_pred=grad_pred,
            normals=normals,
            normals_pred=normals_pred,
        )


def render_image(render_fn, rays, rng, chunk=8192):
    """Render all the pixels of an image (in test mode).

  Args:
    render_fn: function, jit-ed render function.
    rays: a `Rays` namedtuple, the rays to be rendered.
    rng: jnp.ndarray, random number generator (used in training mode only).
    chunk: int, the size of chunks to render sequentially.

  Returns:
    rgb: jnp.ndarray, rendered color image.
    disp: jnp.ndarray, rendered disparity image.
    acc: jnp.ndarray, rendered accumulated weights per pixel.
  """
    height, width = rays[0].shape[:2]
    num_rays = height * width
    rays = utils.namedtuple_map(lambda r: r.reshape((num_rays, -1)), rays)

    host_id = jax.host_id()
    results = []
    for i in range(0, num_rays, chunk):
        # pylint: disable=cell-var-from-loop
        chunk_rays = utils.namedtuple_map(lambda r: r[i:i + chunk], rays)
        chunk_size = chunk_rays[0].shape[0]
        rays_remaining = chunk_size % jax.device_count()
        if rays_remaining != 0:
            padding = jax.device_count() - rays_remaining
            chunk_rays = utils.namedtuple_map(
                lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode='edge'), chunk_rays)
        else:
            padding = 0
        # After padding the number of chunk_rays is always divisible by
        # host_count.
        rays_per_host = chunk_rays[0].shape[0] // jax.host_count()
        start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
        chunk_rays = utils.namedtuple_map(lambda r: utils.shard(r[start:stop]),
                                          chunk_rays)
        chunk_results = render_fn(rng, chunk_rays)[-1]
        results.append([utils.unshard(x[0], padding) for x in chunk_results])
        # pylint: enable=cell-var-from-loop
    rgb, distance, acc = [jnp.concatenate(r, axis=0) for r in zip(*results)]
    rgb = rgb.reshape((height, width, -1))
    distance = distance.reshape((height, width))
    acc = acc.reshape((height, width))
    return (rgb, distance, acc)
