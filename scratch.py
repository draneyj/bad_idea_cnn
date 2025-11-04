import jax

accelerator_devices = jax.devices()
  for accelerator in accelerators.flat:
    accelerator_devices.append(accelerator)
  try:
    hash_obj.update(
        xla_client.get_topology_for_devices(accelerator_devices).serialize()
    )