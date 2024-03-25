using GeophysicalFlows, Random, Printf, CUDA, JLD2
using LinearAlgebra: mul!, ldiv!
parsevalsum = FourierFlows.parsevalsum

dev = GPU()     # Device (CPU/GPU)

# ## Numerical, domain, and simulation parameters

 n, L  = 512, 2π             # grid resolution and domain length
 ν, nν = 0.01, 1             # viscosity coefficient and hyperviscosity order
 μ, nμ = 0.1, 0              # linear drag coefficient
    dt = 0.0002              # timestep
nsteps = 20000               # total number of steps
 nsubs = 100                 # number of steps between each plot

# ## Forcing

forcing_wavenumber = 5.0 * 2π/L  # the forcing wavenumber
amp = -10.0*forcing_wavenumber

grid = TwoDGrid(dev; nx=n, Lx=L)
K = @. sqrt(grid.Krsq)             # a 2D array with the total wavenumber

force = device_array(dev)([amp*cos(forcing_wavenumber*j) for i in grid.x, j in grid.y])
force_h = device_array(dev)(Complex.(zeros(Int(n/2)+1, Int(n))))
mul!(force_h, grid.rfftplan, deepcopy(force))

@CUDA.allowscalar force_h[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average

function calcF!(Fh, sol, t, clock, vars, params, grid) 
  Fh .= force_h  
  return nothing
end

# ## Problem setup

prob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν=ν, nν=nν, μ=μ, nμ=nμ, dt=dt, stepper="ETDRK4",
                                calcF=calcF!, stochastic=true)

sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = grid.x, grid.y

calcF!(vars.Fh, sol, 0.0, clock, vars, params, grid)

# ## Setting initial conditions

dev2 = CPU()
grid2 = TwoDGrid(dev2; nx=n, Lx=L)

data = jldopen("/storage/p2/parfenyev/2d_turbulence/pinn/get_data/alpha_0.1/alpha_01_v1.jld2")
in_cond_h = data["snapshots/sol/1000000"]
in_cond = zeros(Int(n),Int(n))
ldiv!(in_cond, grid2.rfftplan, deepcopy(in_cond_h))
close(data)

TwoDNavierStokes.set_ζ!(prob, device_array(dev)(in_cond))

# ## Diagnostics

E  = Diagnostic(TwoDNavierStokes.energy, prob, nsteps=nsteps) # energy
Z = Diagnostic(TwoDNavierStokes.enstrophy, prob; nsteps=nsteps)
W = Diagnostic(TwoDNavierStokes.energy_work, prob, nsteps=nsteps) # energy work input by forcing
diags = [E, Z, W] # a list of Diagnostics passed to `stepforward!` will  be updated every timestep.

# ## Output

filepath = "/storage/p2/parfenyev/2d_turbulence/pinn/get_data/alpha_0.1"
filename = joinpath(filepath, "alpha_01_v2.jld2")
if isfile(filename); rm(filename); end

get_sol(prob) = Array(prob.sol) # extracts variables
out = Output(prob, filename, (:sol, get_sol), 
    (:E, TwoDNavierStokes.energy), (:Z, TwoDNavierStokes.enstrophy), (:W, TwoDNavierStokes.energy_work))
saveproblem(out)


# Finally, we time-step the `Problem` forward in time.

startwalltime = time()

for j = 0:round(Int, nsteps / nsubs)
  cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
  log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Z: %.4f, W: %.4f, walltime: %.2f min",
        clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i], W.data[W.i], (time()-startwalltime)/60)
  println(log)
  
  stepforward!(prob, diags, nsubs)
  TwoDNavierStokes.updatevars!(prob)
  saveoutput(out)
end