using Gen
using LinearAlgebra
using Statistics

"""
Shawn Nordstrom
12/01/2025
"""

const N_GRID      = 12 
const DX          = 0.03  
const DT          = 5e-4   
const SUBSTEPS    = 20
const GRAVITY_Y   = -9.8
const Y_FLOOR     = 0.1
const MASS        = 1.0
const RESTITUTION = 0.3  
const K_MIN = 10.0
const K_MAX = 40.0
const C_MIN = 1.0
const C_MAX = 5.0
const NOISE_SCALE = 0.05


function simulate_soft_square(k::Float64, c::Float64, T::Int)
    n = N_GRID

    pos   = Array{Float64,3}(undef, n, n, 2)
    vel   = zeros(Float64, n, n, 2)
    force = zeros(Float64, n, n, 2)

    for i in 1:n
        for j in 1:n
            x = 0.5 + (i - (n + 1) / 2) * DX
            y = 0.6 + (j - (n + 1) / 2) * DX
            pos[i, j, 1] = x
            pos[i, j, 2] = y
        end
    end

    traj = Array{Float64,4}(undef, T, n, n, 2)

    function add_spring!(force, pos, i1, j1, i2, j2, rest_len)
        p1x = pos[i1, j1, 1]; p1y = pos[i1, j1, 2]
        p2x = pos[i2, j2, 1]; p2y = pos[i2, j2, 2]
        dx_ = p2x - p1x
        dy_ = p2y - p1y
        L = sqrt(dx_^2 + dy_^2) + 1e-6
        dirx = dx_ / L
        diry = dy_ / L
        extension = L - rest_len
        Fx = -k * extension * dirx
        Fy = -k * extension * diry
        force[i1, j1, 1] += Fx
        force[i1, j1, 2] += Fy
        force[i2, j2, 1] -= Fx
        force[i2, j2, 2] -= Fy
    end

    for t in 1:T
        for s in 1:SUBSTEPS
            for i in 1:n, j in 1:n
                force[i, j, 1] = 0.0
                force[i, j, 2] = MASS * GRAVITY_Y
            end

            # structural springs (4-neighbor)
            for i in 1:n, j in 1:n
                if i + 1 <= n
                    add_spring!(force, pos, i, j, i+1, j, DX)
                end
                if j + 1 <= n
                    add_spring!(force, pos, i, j, i, j+1, DX)
                end
            end

            # shear springs (diagonals)
            for i in 1:n, j in 1:n
                if i + 1 <= n && j + 1 <= n
                    add_spring!(force, pos, i, j, i+1, j+1, DX * sqrt(2.0))
                end
                if i + 1 <= n && j - 1 >= 1
                    add_spring!(force, pos, i, j, i+1, j-1, DX * sqrt(2.0))
                end
            end

            # bending springs (two apart)
            for i in 1:n, j in 1:n
                if i + 2 <= n
                    add_spring!(force, pos, i, j, i+2, j, 2*DX)
                end
                if j + 2 <= n
                    add_spring!(force, pos, i, j, i, j+2, 2*DX)
                end
            end

            # damping
            for i in 1:n, j in 1:n
                force[i, j, 1] += -c * vel[i, j, 1]
                force[i, j, 2] += -c * vel[i, j, 2]
            end

            # integration (semi-implicit euler)
            for i in 1:n, j in 1:n
                ax = force[i, j, 1] / MASS
                ay = force[i, j, 2] / MASS
                vel[i, j, 1] += ax * DT
                vel[i, j, 2] += ay * DT
                pos[i, j, 1] += vel[i, j, 1] * DT
                pos[i, j, 2] += vel[i, j, 2] * DT

                # floor collision
                if pos[i, j, 2] < Y_FLOOR
                    pos[i, j, 2] = Y_FLOOR
                    if vel[i, j, 2] < 0.0
                        vel[i, j, 2] *= -RESTITUTION
                    end
                end
            end
        end

        for i in 1:n, j in 1:n
            traj[t, i, j, 1] = pos[i, j, 1]
            traj[t, i, j, 2] = pos[i, j, 2]
        end
    end

    return traj
end


function center_of_mass(traj::Array{Float64,4})
    T, n, _, _ = size(traj)
    com = zeros(Float64, T, 2)
    N = n * n
    for t in 1:T
        sx = 0.0
        sy = 0.0
        for i in 1:n, j in 1:n
            sx += traj[t, i, j, 1]
            sy += traj[t, i, j, 2]
        end
        com[t, 1] = sx / N
        com[t, 2] = sy / N
    end
    return com
end

function vertical_extent(traj::Array{Float64,4})
    T, n, _, _ = size(traj)
    ext = zeros(Float64, T)
    for t in 1:T
        ymin = Inf
        ymax = -Inf
        for i in 1:n, j in 1:n
            y = traj[t, i, j, 2]
            ymin = min(ymin, y)
            ymax = max(ymax, y)
        end
        ext[t] = ymax - ymin
    end
    return ext
end

function soft_square_features(traj::Array{Float64,4})
    T, n, _, _ = size(traj)
    dt_frame = SUBSTEPS * DT

    com = center_of_mass(traj)             # (T, 2)
    ext = vertical_extent(traj)            # (T,)

    if T < 2
        vel = zeros(Float64, 1, 2)
    else
        vel = diff(com, dims=1) ./ dt_frame   # (T-1, 2)
    end
    vy = vec(vel[:, 2])
    speed = vec(sqrt.(vel[:,1].^2 .+ vel[:,2].^2))

    # max compression ratio
    initial_extent = ext[1]
    compression_ratio = ext ./ (initial_extent + 1e-6)
    max_compression_ratio = minimum(compression_ratio)  # ≤ 1

    # settling time
    com_y = com[:, 2]
    final_y = com_y[end]
    final_vy = (length(vy) > 0) ? vy[end] : 0.0

    eps_y = 0.01
    eps_v = 0.05

    settling_index = T
    for t in 1:T
        ok_y = true
        ok_v = true
        for τ in t:T
            if abs(com_y[τ] - final_y) ≥ eps_y
                ok_y = false
                break
            end
        end
        for τ in max(t-1, 1):(T-1)
            if abs(vy[τ] - final_vy) ≥ eps_v
                ok_v = false
                break
            end
        end
        if ok_y && ok_v
            settling_index = t
            break
        end
    end
    settling_time = (settling_index - 1) * dt_frame

    # max COM speed
    max_speed = (length(speed) > 0) ? maximum(speed) : 0.0

    # bounce count
    bounce_count = 0
    for t in 2:length(vy)
        if vy[t-1] < 0.0 && vy[t] >= 0.0
            bounce_count += 1
        end
    end

    return Float64[
        max_compression_ratio,
        settling_time,
        max_speed,
        float(bounce_count),
    ]
end


@gen function soft_square_model(T::Int, noise_scale::Float64)
    # Latent parameters θ = (k, c)
    k ~ uniform(K_MIN, K_MAX)
    c ~ uniform(C_MIN, C_MAX)

    traj = simulate_soft_square(k, c, T)
    phi = soft_square_features(traj)
    D = length(phi)

    for d in 1:D
        {(:phi, d)} ~ normal(phi[d], noise_scale)
    end

    return (k = k, c = c)
end

function posterior_samples_soft_square(phi_obs::AbstractVector{<:Real};
                                       T::Int,
                                       noise_scale::Float64 = NOISE_SCALE,
                                       num_particles::Int = 100)
    D = length(phi_obs)

    constraints = choicemap()
    for d in 1:D
        constraints[(:phi, d)] = phi_obs[d]
    end

    traces, log_weights = importance_sampling(
        soft_square_model,
        (T, noise_scale),
        constraints,
        num_particles
    )

    maxlog = maximum(log_weights)
    w_unnorm = exp.(log_weights .- maxlog)
    w = w_unnorm ./ sum(w_unnorm)

    ks = Float64[]
    cs = Float64[]
    for tr in traces
        push!(ks, get_choice(tr, :k))
        push!(cs, get_choice(tr, :c))
    end

    return (ks = ks, cs = cs, weights = w)
end

function map_estimate(phi_obs::AbstractVector{<:Real};
                      T::Int,
                      noise_scale::Float64 = NOISE_SCALE,
                      num_particles::Int = 100)
    samples = posterior_samples_soft_square(phi_obs;
                                            T = T,
                                            noise_scale = noise_scale,
                                            num_particles = num_particles)
    ks, cs, w = samples.ks, samples.cs, samples.weights
    idx = argmax(w)
    return (k_hat = ks[idx], c_hat = cs[idx])
end

if abspath(PROGRAM_FILE) == @__FILE__
    k_true = 25.0
    c_true = 3.0
    T = 60

    @info "Simulating ground-truth trajectory..." k_true c_true T
    traj_true = simulate_soft_square(k_true, c_true, T)
    phi_obs = soft_square_features(traj_true)

    @info "Running importance sampling inference in Gen..."
    est = map_estimate(phi_obs; T = T, noise_scale = NOISE_SCALE, num_particles = 200)

    @info "True parameters" k_true c_true
    @info "MAP estimate" est.k_hat est.c_hat
end