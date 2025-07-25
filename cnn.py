import jax
import jax.numpy as jnp
from functools import partial
import numpy as np


@jax.jit
def coord_to_corners(scaled_r):
    r = scaled_r
    r1 = r - jnp.floor(r)
    r2 = jnp.ceil(r) - r
    rr = 1 - jnp.stack([r1,r2])
    return jnp.array([[[[rr[x][0] * rr[y][1] * rr[z][2]
                    ] for z in [0, 1]
                    ] for y in [0, 1]
                    ] for x in [0, 1]
                    ])


# @partial(jax.jit, static_argnames=["ngrid", "nspecies"])
def R_to_grids(scaled_R, species, scaled_box, ngrid, nspecies=3):
    nx, ny, nz = ngrid
    mapped_positions = jnp.zeros((nx, ny, nz, nspecies))
    # todo: check if there's a more numpy way to do this:
    meshgrid = jnp.array([[[[(x,y,z,s) for s in jnp.arange(nspecies)] for z in jnp.arange(nz)] for y in jnp.arange(ny)] for x in jnp.arange(nx)])
    for s in range(nspecies):
        for r in scaled_R:
            corners = coord_to_corners(r)
            fr = jnp.floor(r)
            print(f"r: {fr}")
            for x in [0, 1]:
                for y in [0, 1]:
                    for z in [0, 1]:
                        print(f"r: {fr + jnp.array([x,y,z])}")
                        print(f"r mod: {jnp.mod(fr + jnp.array([x, y, z]), jnp.array([nx, ny, nz]))}")
                        mask = jnp.all(meshgrid[...,:3] == (jnp.mod(fr + jnp.array([x, y, z]), jnp.array([nx, ny, nz]))), axis=-1)
                        mask &= (meshgrid[...,3] == s)
                        print(jnp.where(mask))
                        mapped_positions += jnp.where(mask, corners[x, y, z], 0)
    return mapped_positions


class CNN():
    def __init__(self, kernel_sizes: list, nfeatures: list, nspecies: int = 3):
        self.kernel_sizes = kernel_sizes
        self.nfeatures = [nspecies] + nfeatures
        # print(f"Kernel sizes: {self.kernel_sizes}, Feature sizes: {self.nfeatures}, Number of species: {nspecies}")
        if len(kernel_sizes) != len(nfeatures):
            raise ValueError("Number of kernel sizes must match number of feature layers")
        if np.any(np.array(kernel_sizes) % 2 == 0):
            raise ValueError("All kernel sizes must be odd")

    @partial(jax.jit, static_argnames=["self", "kernel_size"])
    def custom_semiperiodic_conv(self, kernels, input, kernel_size=5):
        """Custom convolution that handles periodic boundary conditions for all but the last dimension."""
        pad_size = kernel_size // 2
        periodic_input_shape = tuple([s + 2 * pad_size for s in input.shape[1:-1]])
        periodic_input_shape = (input.shape[0],) + periodic_input_shape + (input.shape[-1],)
        periodic_input = jnp.zeros(periodic_input_shape)
        pad_index_map = {-1: slice(None, pad_size), 0: slice(pad_size, -pad_size), 1: slice(-pad_size, None)}
        input_index_map = {-1: slice(-pad_size, None), 0: slice(None, None), 1: slice(None, pad_size)}
        for pad_index_x in [-1, 0, 1]:
            for pad_index_y in [-1, 0, 1]:
                for pad_index_z in [-1, 0, 1]:
                    periodic_input = periodic_input.at[
                        :,
                        pad_index_map[pad_index_x],
                        pad_index_map[pad_index_y],
                        pad_index_map[pad_index_z],
                        :,
                    ].set(input[
                        :,
                        input_index_map[pad_index_x], 
                        input_index_map[pad_index_y], 
                        input_index_map[pad_index_z],
                        :,
                    ])
        return jax.lax.conv_general_dilated(
            periodic_input,
            kernels,
            window_strides=(1, 1, 1),
            padding="VALID",
            dimension_numbers=('NHWDC', 'OHWDI', 'NHWDC'),
        )

    def setup_kernels(self, key):
        kernels = []
        for i in range(len(self.kernel_sizes)):
            kernel_shape = (self.nfeatures[i + 1], self.kernel_sizes[i], self.kernel_sizes[i], self.kernel_sizes[i], self.nfeatures[i])
            kernel = jax.random.normal(key, shape=kernel_shape)
            kernels.append(kernel)
        return kernels

    @partial(jax.jit, static_argnames=["self"])
    def __call__(self, kernels, inputs):
        # add batch dimension for convolution
        x = jnp.reshape(inputs, (1,) + inputs.shape)
        for kernel, kernel_size in zip(kernels, self.kernel_sizes):
            x = jax.nn.gelu(x)
            x = self.custom_semiperiodic_conv(kernel, x, kernel_size)
        return jnp.sum(x)


@partial(jax.jit, static_argnames=["ngrid", "nspecies", "network"])
def energy(kernels, network, scaled_R, species, scaled_box, ngrid, nspecies=3):
    mapped_positions = R_to_grids(scaled_R, species, scaled_box, ngrid, nspecies=3)
    return network(kernels, mapped_positions)


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