## Content

`./get_data/script.jl` -- The script is used to integrate the Navier-Stokes equation from some random initial state, and after a transient process the system reaches the statistical steady-state.

`./get_data/script2.jl` -- Now the system is already in the statistical steady-state and we are generating the desired amount of data.

`./processing.ipynb` -- Data visualization and calculation of the velocity and pressure from the stored vorticity exploiting periodic boundary conditions.

`./video.mp4` -- Vorticity dynamics for the generated data.

All scripts in this folder are written in [Julia](https://julialang.org/). You can also [download the data](https://parfenyev.itp.ac.ru/data/2d-turb-PINN/) used for PINN training and validation in HDF5 format.
