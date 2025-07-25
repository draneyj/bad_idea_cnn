import numpy as np
import optax
import jax
import jax.numpy as jnp

opt = optax.adam(1e-3)
k = np.random.random((10, 10))
print(k[0,0])

state = opt.init(k)

def loss(k):
    return jnp.sum(jnp.square(k - 0.55555))


for i in range(10000):
    thisloss, grads = jax.value_and_grad(loss, 0)(k)
    print(f"LOSS: {thisloss}")
    updates, state = opt.update(grads, state)
    k = optax.apply_updates(k, updates)
    print(k[0,0])
