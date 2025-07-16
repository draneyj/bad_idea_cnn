import jax
import jax.numpy as jnp
from functools import partial
import numpy as np

def coord_to_corners(scaled_r):
    r = scaled_r
    r1 = r - jnp.floor(r)
    r2 = jnp.ceil(r) - r
    rr = jnp.stack([r1,r2])
    return jnp.array([[[[rr[x][0] * rr[y][1] * rr[z][2]
                    ] for z in [0, 1]
                    ] for y in [0, 1]
                    ] for x in [0, 1]
                    ])


@partial(jax.jit, static_argnames=["ngrid", "nspecies"])
def R_to_grids(scaled_R, species, scaled_box, ngrid, nspecies=3):
    nx, ny, nz = ngrid
    mapped_positions = jnp.zeros((nx, ny, nz, nspecies))
    # todo: check if there's a more numpy way to do this:
    meshgrid = jnp.array([[[[(x,y,z,s) for s in jnp.arange(nspecies)] for z in jnp.arange(nz)] for y in jnp.arange(ny)] for x in jnp.arange(nx)])
    for s in range(nspecies):
        for r in scaled_R:
            corners = coord_to_corners(r)
            fr = jnp.floor(r)
            for x in [0, 1]:
                for y in [0, 1]:
                    for z in [0, 1]:
                        mask = jnp.all(meshgrid[:,:,:,:,:3] == (jnp.mod(fr + jnp.array([x, y, z]), jnp.array([nx, ny, nz]))), axis=-1)
                        mask &= (meshgrid[:,:,:,:,3] == s)
                        mapped_positions += jnp.where(mask, corners[x, y, z], 0)
    return mapped_positions


def init_random_params(kernel_sizes, nspecies):
    layers = []
    for n in kernel_sizes:
        layers.append(np.random.normal(0,1,(n,n,n,nspecies)))
    return jnp.array(layers)


def periodic_convolve(x,kernel):
    pass


@jax.jit
def cnn(layers, mapped_positions):
    x = periodic_convolve(mapped_positions, layers[0])
    for layer in layers[1:]:
        x = jax.nn.gelu(x)
        x = periodic_convolve(x, layer)
    return jnp.sum(x)


@partial(jax.jit, static_argnames=["ngrid", "nspecies"])
def energy(layers, scaled_R, species, scaled_box, ngrid, nspecies=3):
    mapped_positions = R_to_grids(scaled_R, species, scaled_box, ngrid, nspecies=3)
    return cnn(layers, mapped_positions)


def random_transform(scaled_R, forces):
    # rotate
    rotate_z = np.random.random() * np.pi
    rotate_y = np.random.random() * np.pi
    
    rotation = np.array([
        [np.cos(rotate_z), np.cos(rotate_z), 0,],
        [np.sin(rotate_z), -np.sin(rotate_z), 0,],
        [0, 0, 1,],
    ])
    rotation = rotation @ np.array([
        [np.cos(rotate_y), 0, np.cos(rotate_y),],
        [0, 1, 0,],
        [np.sin(rotate_y), 0, -np.sin(rotate_y),],
    ])
    translation = np.random.random(3)

    return rotation @ scaled_R + translation, rotation @ forces