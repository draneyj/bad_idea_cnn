import os
os.environ["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler,jax._src.lru_cache"
os.environ["JAX_LOGGING_LEVEL"] = "DEBUG"
os.environ["JAX_RAISE_PERSISTENT_CACHE_ERRORS"] = "True"
import jax
import jax.numpy as jnp
import cnn
import numpy as np
import pickle
import optax
from functools import partial

np.random.seed(72099)

jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_explain_cache_misses", True)

# set up network parameters
a = 3.577678
scale = a / 2
nspecies = 3
kernel_sizes = (7, 5, 5, 3)
nfeatures = [16, 16, 16, 1]
key = jax.random.PRNGKey(0)
kernels = cnn.setup_kernels(kernel_sizes, nfeatures, key)

# load up on training data
rows = pickle.load(open("CHO_dataset.pkl", "rb"))
rows = rows['structures']


def pad_R_species_forces(R, species, forces, pad_to):
    if len(species) >= pad_to:
        return R, species, forces
    new_R = np.zeros((pad_to, 3))
    new_F = np.zeros((pad_to, 3))
    new_species = np.full(pad_to, -1, dtype=jnp.int32)
    new_R[:R.shape[0], :] = R
    new_species[:species.shape[0]] = species
    new_F[:forces.shape[0], :] = forces
    return new_R, new_species, new_F


@partial(jax.jit, static_argnames=["nx", "ny", "nz", "nspecies", "kernel_sizes"])
def scale_and_energy(kernels, positions, scaled_box, nx, ny, nz, species, kernel_sizes, nspecies=3):
    scaled_R = positions / scale 
    return cnn.energy(kernels, kernel_sizes, scaled_R, species, scaled_box, nx, ny, nz, nspecies=nspecies)

energy_and_negforce = jax.value_and_grad(scale_and_energy, 1)

@partial(jax.jit, static_argnames=["nx", "ny", "nz", "kernel_sizes"])
def loss_function(kernels, positions, scaled_box, nx, ny, nz, species, kernel_sizes, true_energy, true_forces, e_weight=1.0, f_weight=1.0):
    energy, neg_forces = energy_and_negforce(kernels, positions, scaled_box, nx, ny, nz, species, kernel_sizes)
    energy_error = (true_energy - energy)**2
    num_forces = jnp.sum(species >= 0)
    force_error = jnp.sum(jnp.where(species >= 0, jnp.sum((neg_forces + true_forces)**2, axis=-1), 0)) / num_forces
    return e_weight * energy_error + f_weight * force_error

vg_loss_function = jax.jit(jax.value_and_grad(loss_function, 0), static_argnames=["nx", "ny", "nz", "kernel_sizes"])

# set up learning parameters
numsteps = 1
lr = 1e-3
smap = {"C":0, "H":1, "O":2}

# set up simple train validation split
indicies = np.arange(len(rows))
np.random.shuffle(indicies)
validation_indicies = indicies[:len(rows) // 5]
train_indicies = indicies[len(rows) // 5:]
train_rows = [rows[i] for i in train_indicies]
validation_rows = [rows[i] for i in validation_indicies]
num_validation = len(validation_rows)
min_validation_loss = np.inf

# initialize optimizer
opt = optax.adam(lr)
state = opt.init(kernels)

for i in range(numsteps):
    print(f"========== epoch {i} ==========", flush=True)
    for j, row in enumerate(train_rows):
        positions = row['coordinates']
        orth_matrix = row['orth_matrix']
        species = jnp.array([smap[s] for s in row['species']])
        true_energy = row['energy']
        true_forces = row['forces']
        positions, species, true_forces = pad_R_species_forces(positions, species, true_forces, 256)
        scaled_box = np.round(np.array([[0, orth_matrix[0,0]], [0, orth_matrix[1,1]], [0, orth_matrix[2,2]]]) / scale)
        nx = int(scaled_box[0][1] - scaled_box[0][0])
        ny = int(scaled_box[1][1] - scaled_box[1][0])
        nz = int(scaled_box[2][1] - scaled_box[2][0])

        compiled = vg_loss_function.lower(kernels, positions, scaled_box, nx, ny, nz, species, kernel_sizes, true_energy, true_forces).compiler_ir('hlo')
        print(compiled.as_hlo_text(), flush=True)
        
        exit()

        print(f"nx: {nx}, ny: {ny}, nz: {nz}", flush=True)
        loss, grad = vg_loss_function(kernels, positions, scaled_box, nx, ny, nz, species, kernel_sizes, true_energy, true_forces)
        updates, state = opt.update(grad, state)
        kernels = optax.apply_updates(kernels, updates)
        print(f"training row {j}: {loss}", flush=True)

    print("validating ...")
    for row in validation_rows:
        positions = row['coordinates']
        orth_matrix = row['orth_matrix']
        species = jnp.array([smap[s] for s in row['species']])
        true_energy = row['energy']
        true_forces = row['forces']
        vloss += loss_function(kernels, positions, orth_matrix, species, kernel_sizes, true_energy, true_forces) / num_validation
    print(f"validation loss is {vloss}")
    if vloss < min_validation_loss:
        print("new best params!")
        pickle.dump(kernels, open("min_kernels.pkl", "wb"))

print("done.")