from flax import struct

@struct.dataclass
class Model:
  params: Any
  # use pytree_node=False to indicate an attribute should not be touched
  # by Jax transformations.
  apply_fn: FunctionType = struct.field(pytree_node=False)

  def __apply__(self, *args):
    return self.apply_fn(*args)

model = Model(params, apply_fn)

model.params = params_b  # Model is immutable. This will raise an error.
model_b = model.replace(params=params_b)  # Use the replace method instead.

# This class can now be used safely in Jax to compute gradients w.r.t. the
# parameters.
model = Model(params, apply_fn)
model_grad = jax.grad(some_loss_fn)(model)