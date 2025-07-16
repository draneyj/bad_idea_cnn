import jax
import jax.numpy as jnp
import cnn
import numpy as np

N = 10
nspecies = 1
species = jnp.full(0, N)
box = jnp.array([[0, 10], [0, 10], [0, 10]])
R = jnp.array(np.random.random((N, 3))) * box[:, 2]
print(R)
gridspace = .5
scaled_box = jnp.round(box / gridspace)
scaled_R = R / gridspace

nx = int(scaled_box[0][1] - scaled_box[0][0])
ny = int(scaled_box[1][1] - scaled_box[1][0])
nz = int(scaled_box[2][1] - scaled_box[2][0])
ngrid = ny, ny, nz

gmps = cnn.R_to_grids(scaled_R, species, scaled_box, ngrid, nspecies)
