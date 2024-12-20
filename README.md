# 2d-turb-PINN

Our research is motivated by hydrodynamic experiments and the desire to obtain more information from traditional PIV/PTV measurements of the velocity field. We present a technique that integrates measurement data with physical laws, expressed as partial differential equations, allowing for the simultaneous solution of super-resolution and inference problems. As an example, we consider a two-dimensional turbulent forced fluid flow and, using only sparse and probably noisy data for the velocity field, we reconstruct, the dense velocity and pressure fields in the observation region, infer the driving force, and determine the unknown fluid viscosity and bottom friction coefficient. The suggested technique demonstrates moderate robustness to noise in the measurement data and involves training a physics-informed neural network by minimizing the loss function, which penalizes deviations from the provided data and violations of the Navier-Stokes equation. The developed method extracts additional information from experimental and numerical observations, potentially enhancing the capabilities of PIV/PTV.

For more information, please refer to the following:
- V. Parfenyev, M. Blumenau, I. Nikitin, "Inferring parameters and reconstruction of two-dimensional turbulent flows with physics-informed neural networks", JETP Letters 120, 599-607 (2024); [arXiv:2404.01193](https://arxiv.org/pdf/2404.01193).

You can generate the data for demonstration and training yourself (see `./data_generation/`) or download it from [here](https://parfenyev.itp.ac.ru/data/2d-turb-PINN/) in HDF5 format. The folder `./PINN/saved_models/` contains already trained models. Don't forget to update the file paths in the scripts.

## Acknowledgements

When developing the code, we relied and were inspired by the following repositories:
- https://github.com/maziarraissi/PINNs
- https://github.com/jayroxis/PINNs
- https://github.com/FourierFlows/GeophysicalFlows.jl
