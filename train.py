import jax
import jax.numpy as jnp
import cnn
import numpy as np
import pickle
import optax
from functools import partial

# set up network parameters
a = 3.577678
scale = a / 2
nspecies = 3
kernel_sizes = [7, 5, 5, 3]
nfeatures = [16, 16, 16, 1]
network = cnn.CNN(kernel_sizes, nfeatures, nspecies)
seed = jax.random.PRNGKey(0)
kernels = network.setup_kernels(seed)

# load up on training data
rows = pickle.load(open("CHO_dataset.pkl", "rb"))
rows = rows['structures']

@partial(jax.jit, static_argnames=["nx", "ny", "nz", "nspecies"])
def energy(kernels, positions, scaled_box, nx, ny, nz, species, nspecies=3):
    scaled_R = positions / scale 
    return cnn.energy(kernels, network, scaled_R, species, scaled_box, nx, ny, nz, nspecies=nspecies)

energy_and_negforce = jax.jit(jax.value_and_grad(energy, 1), static_argnames=["nx", "ny", "nz", "nspecies"])

@partial(jax.jit, static_argnames=["nx", "ny", "nz"])
def loss_function(kernels, positions, scaled_box, nx, ny, nz, species, true_energy, true_forces, e_weight=1.0, f_weight=1.0):
    energy, neg_forces = energy_and_negforce(kernels, positions, scaled_box, nx, ny, nz, species)
    energy_error = (true_energy - energy)**2
    force_error = jnp.sum((neg_forces + true_forces)**2)
    return e_weight * energy_error + f_weight * force_error

# set up learning parameters
numsteps = 100
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
_, state = opt.init(kernels)

for i in range(numsteps):
    print(f"========== epoch {i} ==========")
    for j, row in enumerate(train_rows):
        positions = row['coordinates']
        orth_matrix = row['orth_matrix']
        species = jnp.array([smap[s] for s in row['species']])
        true_energy = row['energy']
        true_forces = row['forces']
        scaled_box = np.round(np.array([[0, orth_matrix[0,0]], [0, orth_matrix[1,1]], [0, orth_matrix[2,2]]]) / scale)
        nx = int(scaled_box[0][1] - scaled_box[0][0])
        ny = int(scaled_box[1][1] - scaled_box[1][0])
        nz = int(scaled_box[2][1] - scaled_box[2][0])

        print(f"nx: {nx}, ny: {ny}, nz: {nz}")
        loss, grad = jax.value_and_grad(loss_function, 0)(kernels, positions, scaled_box, nx, ny, nz, species, true_energy, true_forces)
        updates, state = opt.update(grads, state)
        kernels = optax.apply_updates(kernels, updates)
        print(f"training row {j}: {loss}")

    print("validating ...")
    for row in validation_rows:
        positions = row['coordinates']
        orth_matrix = row['orth_matrix']
        species = jnp.array([smap[s] for s in row['species']])
        true_energy = row['energy']
        true_forces = row['forces']
        vloss += loss_function(kernels, positions, orth_matrix, species, true_energy, true_forces) / num_validation
    print(f"validation loss is {vloss}")
    if vloss < min_validation_loss:
        print("new best params!")
        pickle.dump(kernels, open("min_kernels.pkl", "wb"))

print("done.")