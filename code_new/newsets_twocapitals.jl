#=============================================================================#
#  Economy with TWO CAPITAL STOCKS
#
#  Author: Balint Szoke
#  Date: Sep 2018
#=============================================================================#

using Optim
using Roots
using NPZ
using Distributed

#==============================================================================#
# SPECIFICATION:
#==============================================================================#
#symmetric_returns    = parse(Int64, ARGS[1])
#state_dependent_xi   = parse(Int64, ARGS[2])
#optimize_over_ell    = parse(Int64, ARGS[3])
#compute_irfs         = parse(Int64, ARGS[4])


symmetric_returns    = 0
state_dependent_xi   = 2
optimize_over_ell    = 0
compute_irfs         = 0                    # need to start julia with "-p 5"

if compute_irfs == 1
    @everywhere include("newsets_utils.jl")
elseif compute_irfs ==0
    include("newsets_utils.jl")
end

println("=============================================================")
if symmetric_returns == 1
    println(" Economy with two capital stocks: SYMMETRIC RETURNS          ")
    if state_dependent_xi == 0
        println(" No tilting (xi is NOT state dependent)                      ")
        filename = (compute_irfs==0) ? "model_sym_HS.npz" : "model_sym_HS_p.npz";
    elseif state_dependent_xi == 1
        println(" With tilting (change in kappa)                        ")
        filename = (compute_irfs==0) ? "model_sym_HSHS.npz" : "model_sym_HSHS_p.npz";
    elseif state_dependent_xi == 2
        println(" With tilting (change in beta)                        ")
        filename = (compute_irfs==0) ? "model_sym_HSHS2.npz" : "model_sym_HSHS2_p.npz";
    end
elseif symmetric_returns == 0
    println(" Economy with two capital stocks: ASYMMETRIC RETURNS         ")
    if state_dependent_xi == 0
        println(" No tilting (xi is NOT state dependent)                      ")
        filename = (compute_irfs==0) ? "model_asym_HS.npz" : "model_asym_HS_p.npz";
    elseif state_dependent_xi == 1
        println(" With tilting (change in kappa)                        ")
        filename = (compute_irfs==0) ? "model_asym_HSHS.npz" : "model_asym_HSHS_p.npz";
    elseif state_dependent_xi == 2
        println(" With tilting (change in beta)                        ")
        filename = (compute_irfs==0) ? "model_asym_HSHS2.npz" : "model_asym_HSHS2_p.npz";
    end
end

#==============================================================================#
#  PARAMETERS
#==============================================================================#
delta = .002;

# (0) Single capital economy
alpha_c_hat = .484;
beta_hat = 1.0;
sigma_c = [.477, .0];

#===========================  CALIBRATION  ====================================#
# consumption_investment = 3.1;
#A_1cap, phi_1cap, alpha_k_hat, investment_capital = calibration2(15.,
#                                             consumption_investment,
#                                             alpha_c_hat, delta, sigma_c)
# A_1cap, phi_1cap, alpha_k_hat = calibration3(investment_capital,
#                                   consumption_investment,
#                                   alpha_c_hat, delta, sigma_c)
#

A_1cap = .05
phi_1cap = 28.
investment_capital, consumption_investment, alpha_k_hat = calibration3(phi_1cap,
                                            A_1cap, delta, alpha_c_hat, sigma_c)

println("  Calibrated values: A:", A_1cap,
        "  phi_1cap: ", phi_1cap,
        "  alpha_k : ", alpha_k_hat,
        "  C/I : ", consumption_investment,
        "  I/K : ", investment_capital)
println("=============================================================")
#==============================================================================#

# (1) Baseline model
alpha_z_hat = .0;
kappa_hat = .014;
zbar = alpha_z_hat/kappa_hat;
sigma_z_1cap = [.011, .025];

sigma_z =  [.011*sqrt(.5)   , .011*sqrt(.5)   , .025];


if symmetric_returns == 1

    beta2_hat = beta1_hat = 0.5;

    # (2) Technology
    phi2 = phi1 = phi_1cap;
    A2 = A1 = A_1cap;

    if state_dependent_xi == 0
        # Constant tilting function
        scale = 1.754;
        alpha_k2_hat = alpha_k1_hat = alpha_k_hat;

        # Worrisome model
        alpha_z_tilde  = -.005
        kappa_tilde    = kappa_hat;
        alpha_k1_tilde = alpha_k1_hat
        beta1_tilde    = beta1_hat
        alpha_k2_tilde = alpha_k2_hat
        beta2_tilde    = beta2_hat

        ell_star = 0.055594409575544096

    elseif state_dependent_xi == 1
        # State-dependent tilting function (fixed kappa, alpha targets q)
        scale = 1.62
        alpha_k2_hat = alpha_k1_hat = alpha_k_hat;

        alpha_z_tilde  = -.00155;
        kappa_tilde    =  .005
        alpha_k1_tilde = alpha_k1_hat
        beta1_tilde    = beta1_hat
        alpha_k2_tilde = alpha_k2_hat
        beta2_tilde    = beta2_hat

        ell_star = 0.13852940062708508

    elseif state_dependent_xi == 2
        # State-dependent tilting function
        scale = 1.568
        alpha_k2_hat = alpha_k1_hat = alpha_k_hat;

        alpha_z_tilde  = -.00155;
        kappa_tilde    = kappa_hat
        alpha_k1_tilde = alpha_k1_hat
        beta1_tilde    = beta1_hat + .1941
        alpha_k2_tilde = alpha_k2_hat
        beta2_tilde    = beta2_hat + .1941

        ell_star = 0.18756641482672026

    end


elseif symmetric_returns == 0

    beta1_hat = 0.0;
    beta2_hat = 0.5;

    # (2) Technology
    phi2 = phi1 = phi_1cap;
    A2 = A1 = A_1cap;

    if state_dependent_xi == 0
        # Constant tilting function
        scale = 1.307
        alpha_k2_hat = alpha_k1_hat = alpha_k_hat;

        # Worrisome model
        alpha_z_tilde  = -.00534;
        kappa_tilde    = kappa_hat;
        alpha_k1_tilde = alpha_k1_hat
        beta1_tilde    = beta1_hat
        alpha_k2_tilde = alpha_k2_hat
        beta2_tilde    = beta2_hat

        ell_star = 0.026320287107624605

    elseif state_dependent_xi == 1
        # State-dependent tilting function (fixed kappa, alpha targets q)
        scale = 1.14
        alpha_k2_hat = alpha_k1_hat = alpha_k_hat + .035; #.034;

        alpha_z_tilde  = -.002325
        kappa_tilde    = .005;
        alpha_k1_tilde = alpha_k1_hat
        beta1_tilde    = beta1_hat;
        alpha_k2_tilde = alpha_k2_hat
        beta2_tilde    = beta2_hat

        ell_star = 0.04226404306515605

    elseif state_dependent_xi == 2
        # State-dependent tilting function (fixed beta1, alpha targets q)
        scale = 1.27
        alpha_k2_hat = alpha_k1_hat = alpha_k_hat

        alpha_z_tilde  = -.002325
        kappa_tilde    = kappa_hat
        alpha_k1_tilde = alpha_k1_hat
        beta1_tilde    = beta1_hat + .194 #.195
        alpha_k2_tilde = alpha_k2_hat
        beta2_tilde    = beta2_hat + .194 #.195

        ell_star = 0.06678494013273199

    end

end

sigma_k1 = [.477*sqrt(scale),               .0,   .0];
sigma_k2 = [.0              , .477*sqrt(scale),   .0];


# (3) GRID
# For analysis
if compute_irfs == 1
    II, JJ = 7001, 501;     # number of r points, number of z points
    rmax = 4.;
    rmin = -rmax;
    zmax = .7;
    zmin = -zmax;
elseif compute_irfs == 0
    II, JJ = 1001, 201;
    rmax =  18.;
    rmin = -rmax       #-25.; #-rmax;
    zmax = 1.;
    zmin = -zmax;
end

# For the optimization (over ell)
II_opt, JJ_opt = 501, 201;     # number of r points, number of z points
rmax_opt = 18.;
rmin_opt = -rmax_opt;
zmax_opt = 1.2;
zmin_opt = -zmax_opt;


# (4) Iteration parameters
maxit = 500;        # maximum number of iterations in the HJB loop
crit  = 10e-6;      # criterion HJB loop
Delta = 1000.;      # delta in HJB algorithm


# Initialize model objects -----------------------------------------------------
baseline = Baseline(alpha_z_hat, kappa_hat, sigma_z_1cap,
                    alpha_c_hat, beta_hat, sigma_c, delta);
baseline1 = Baseline(alpha_z_hat, kappa_hat, sigma_z,
                     alpha_k1_hat, beta1_hat, sigma_k1, delta);
baseline2 = Baseline(alpha_z_hat, kappa_hat, sigma_z,
                     alpha_k2_hat, beta2_hat, sigma_k2, delta);
technology = Technology(A_1cap, phi_1cap);
technology1 = Technology(A1, phi1);
technology2 = Technology(A2, phi2);
model = TwoCapitalEconomy(baseline1, baseline2, technology1, technology2);

worrisome = TwoCapitalWorrisome(alpha_z_tilde, kappa_tilde,
                                alpha_k1_tilde, beta1_tilde,
                                alpha_k2_tilde, beta2_tilde);
worrisome_noR = TwoCapitalWorrisome(alpha_z_hat, kappa_hat,
                                    alpha_k1_hat, beta1_hat,
                                    alpha_k2_hat, beta2_hat);

grid = Grid_rz(rmin, rmax, II, zmin, zmax, JJ);
grid_opt = Grid_rz(rmin_opt, rmax_opt, II_opt, zmin_opt, zmax_opt, JJ_opt);
params = FinDiffMethod(maxit, crit, Delta);

xi0, xi1, xi2 = tilting_function(worrisome, model);


#==============================================================================#
# WITHOUT ROBUSTNESS (indicated by _noR endings)
#==============================================================================#
println(" (1) Compute value function WITHOUT ROBUSTNESS")

@time A_noR, V_noR, val_noR, d1_F_noR, d2_F_noR, d1_B_noR, d2_B_noR,
      h1_F_noR, h2_F_noR, hz_F_noR, h1_B_noR, h2_B_noR, hz_B_noR,
      mu_1_F_noR, mu_1_B_noR, mu_r_F_noR, mu_r_B_noR, mu_z_noR,
      V0, rr, zz, pii, dr, dz = value_function_twocapitals(Inf, model,
                                                           worrisome_noR,
                                                           grid, params,
                                                           symmetric_returns);

g_noR_dist, g_noR = stationary_distribution(A_noR, grid)
mu_1_noR = (mu_1_F_noR + mu_1_B_noR)/2.;
mu_r_noR = (mu_r_F_noR + mu_r_B_noR)/2.;
println("=============================================================")


if symmetric_returns == 0
    if state_dependent_xi == 0
        params.Delta = 14.;
    elseif state_dependent_xi == 1
        params.Delta = 17.;
    elseif state_dependent_xi == 2
        params.Delta = 9.5
    end
end

#==============================================================================#
# WITH ROBUSTNESS
#==============================================================================#
if optimize_over_ell == 1
    println(" (2) ell_star is computed from optimization:                   ")
    #---------------------------------------------------------------------------
    # INITIAL GUESS
    #   when r=-inf/inf, we've single capital case, so we know the value funcs
    #---------------------------------------------------------------------------
    w1_single = Worrisome(worrisome.alpha_z_tilde, worrisome.kappa_tilde,
                          worrisome.alpha_k1_tilde, worrisome.beta1_tilde)
    w2_single = Worrisome(worrisome.alpha_z_tilde, worrisome.kappa_tilde,
                          worrisome.alpha_k2_tilde, worrisome.beta2_tilde)

    res1 = optimize(r-> -value_function(r, zbar, w1_single,
                                        baseline1, technology1)[1], 0.0, 100);
    res2 = optimize(r-> -value_function(r, zbar, w2_single,
                                        baseline2, technology2)[1], 0.0, 100);
    ell = (Optim.minimizer(res1) + Optim.minimizer(res2))/2;


    if (symmetric_returns == 1) && (state_dependent_xi == 2)
        ell_res = optimize(ee -> -value_function_twocapitals(ee, model, worrisome,
                                                             grid_opt, params,
                                                             symmetric_returns)[3],
                           ell*.7, ell*1.3);
        ell_star = Optim.minimizer(ell_res);
        println("  min ell: ", ell*.7,"  ell_star: ", ell_star,"  init ell: ", ell*1.3);
    else
        ell_res = optimize(ee -> -value_function_twocapitals(ee, model, worrisome,
                                                             grid_opt, params,
                                                             symmetric_returns)[3],
                           ell*.5, ell);
        ell_star = Optim.minimizer(ell_res);
        println("  min ell: ", ell*.5,"  ell_star: ", ell_star,"  init ell: ", ell);
    end

elseif optimize_over_ell == 0

    println(" (2) ell_star is given, it is ", ell_star)

end
println("=============================================================")


println(" (3) Compute value function WITH ROBUSTNESS")
A, V, val, d1_F, d2_F, d1_B, d2_B, h1_F, h2_F, hz_F, h1_B, h2_B, hz_B,
        mu_1_F, mu_1_B, mu_r_F, mu_r_B, mu_z, V0, rr, zz, pii, dr, dz =
        value_function_twocapitals(ell_star, model, worrisome,
                                   grid, params, symmetric_returns);
one_pii = 1 .- pii
println("=============================================================")

# Define Policies object
policies  = PolicyFunctions(d1_F, d2_F, d1_B, d2_B,
                            -h1_F/ell_star, -h2_F/ell_star, -hz_F/ell_star,
                            -h1_B/ell_star, -h2_B/ell_star, -hz_B/ell_star);
policies2 = PolicyFunctions(d1_F_noR, d2_F_noR, d1_B_noR, d2_B_noR,
                            -h1_F/ell_star, -h2_F/ell_star, -hz_F/ell_star,
                            -h1_B/ell_star, -h2_B/ell_star, -hz_B/ell_star);

# Construct drift terms under the baseline
mu_1 = (mu_1_F + mu_1_B)/2.;
mu_r = (mu_r_F + mu_r_B)/2.;
# ... under the worst-case model
h1_dist = (policies.h1_F + policies.h1_B)/2.;
h2_dist = (policies.h2_F + policies.h2_B)/2.;
hz_dist = (policies.hz_F + policies.hz_B)/2.;

WCDist_1, WCDist_r, WCDist_z = worstcase_distortion(h1_dist, h2_dist, hz_dist,
                                                    pii, model);
mu_1_wc = mu_1 + WCDist_1;
mu_r_wc = mu_r + WCDist_r;
mu_z_wc = mu_z + WCDist_z;


# Kolmogorov Forward equation under the baseline model
g_dist, g = stationary_distribution(A, grid)

# Kolmogorov Forward equation under the worst-case model
A_wc = Kolmogorov_FinDiff(policies, model, grid, params);
g_wc_dist, g_wc = stationary_distribution(A_wc, grid);

# Kolmogorov Forward eq under the worst-case using non-robust decision rule
A_wc_noR = Kolmogorov_FinDiff(policies2, model, grid, params);
g_wc_noR_dist, g_wc_noR = stationary_distribution(A_wc_noR, grid);

#==============================================================================#
# Approximate relative entropy (need worst-case distribution)
#==============================================================================#
println(" (4) Compute distance between worst-case and baseline")
# local uncertainty prices
h1, h2, hz = -h1_dist, -h2_dist, -hz_dist;

H2 = h1.^2 + h2.^2 + hz.^2;
re = sum(H2 .* g_wc * dz*dr)/2;
q = sqrt(re*2);

# CHERNOFF ENTROPY
#gamma_res = optimize(gamma -> chernoff_objective(gamma, policies, model,
#                                                 grid, params), 1e-5, 1-1e-5);
#chernoff = - Optim.minimum(gamma_res);
#halflife = log(2) / chernoff;

chernoff = 0.0;
halflife = 0.0;

println("    alpha_tilde: ", worrisome.alpha_z_tilde,
        "  kappa_tilde: ", worrisome.kappa_tilde,
        "  q: ", q,
        "  re: ", re,
        "  chernoff: ", chernoff,
        "  halflife: ", halflife);
println("=============================================================")


#==============================================================================#
# Stationary distributions of local uncertainty prices
#==============================================================================#
inner = 1
inI = (inner+1):(II-inner)
inJ = (inner+1):(JJ-inner)

# Single capital
H0, H1 = 0.0, 0.0
# H0, H1 = worst_case(alpha_z_tilde, kappa_tilde, alpha_k2_tilde, beta2_tilde,
#                     baseline, technology)[1:2]
stdev_z_1cap = sqrt(dot(sigma_z_1cap, sigma_z_1cap)/(2*kappa_hat));

h12_vec, h12_density = change_of_variables((h1+h2)/sqrt(2),g,rr,zz, inner)
hz_vec, hz_density   = change_of_variables(hz, g, rr, zz, inner)

# Two capitals economy
#if symmetric_returns == 1
#h12_vec, h12_density = change_of_variables((h1+h2)/sqrt(2),g,rr,zz,inner)
#elseif symmetric_returns == 0
#    h12_vec, h12_density = change_of_variables(h2, g, rr, zz, inner)
#end

#==============================================================================#
# Stationary distributions of consumption
#==============================================================================#
cons_1cap = technology.A - dstar_singlecapital(technology, baseline)
d1_noR = (d1_F_noR + d1_B_noR)/2;
d2_noR = (d2_F_noR + d2_B_noR)/2;
d1 = (policies.d1_F + policies.d1_B)/2;
d2 = (policies.d2_F + policies.d2_B)/2;

cons_noR = one_pii .* (model.t1.A .- d1_noR) + pii .* (model.t2.A .- d2_noR)
cons     = one_pii .* (model.t1.A .- d1) + pii .* (model.t2.A .- d2);

cons_noR_vec, cons_noR_density = change_of_variables(cons_noR, g_noR, rr, zz, inner)
cons_vec, cons_density         = change_of_variables(cons    , g    , rr, zz, inner)
cons_wc_vec, cons_wc_density   = change_of_variables(cons    , g_wc , rr, zz, inner)

# Consumption dynamics
logC_mu_noR, logC_sigma_noR = consumption_dynamics(cons_noR, rr, zz,
                                                   mu_1_noR, mu_r_noR, mu_z_noR,
                                                   model, inner);
logC_mu, logC_sigma         = consumption_dynamics(cons, rr, zz,
                                                   mu_1, mu_r, mu_z,
                                                   model, inner);
logC_mu_wc, logC_sigma_wc   = consumption_dynamics(cons, rr, zz,
                                                   mu_1_wc, mu_r_wc, mu_z_wc,
                                                   model, inner);


logK12_sigma = sqrt.((sigma_k1[1]*one_pii + sigma_k2[1]*pii).^2 +
                     (sigma_k1[2]*one_pii + sigma_k2[2]*pii).^2 +
                     (sigma_k1[3]*one_pii + sigma_k2[3]*pii).^2)[inI, inJ];
logK12_sigma *= .01
drdz = dr*dz


riskfree = zeros(II, JJ);
rf = risk_free_rate(cons, rr, zz, mu_1_wc, mu_r_wc, mu_z_wc, delta, model, inner);
riskfree[inI, inJ] .= rf
riskfree[inner+1:end-inner, 1:inner] .= rf[:, 1]
riskfree[inner+1:end-inner, end-inner:end] .= rf[:, end]
riskfree[1:inner, :] .= riskfree[inner+1:2*inner, :]
riskfree[end-inner+1:end, :] .= riskfree[end-2*inner:end-inner-1, :]

rf_vec, rf_density   = change_of_variables(riskfree, g, rr, zz, inner)

println(" (5) Calibration targets                                     ")
println("    dlogC local mean: ", sum(g[inI,inJ] .* logC_mu * drdz));
println("    dlogC local vol : ", sum(g[inI,inJ] .* logC_sigma * drdz));
println("    dlogK local vol : ", sum(g[inI,inJ] .* logK12_sigma * drdz));
println("    C/I ratio:        ", sum(g .* cons ./ (one_pii .* d1 + pii .* d2))*drdz);
println("    I/K ratio:        ", sum(g .* (one_pii .* d1 + pii .* d2)) * drdz);
println("    dlogC_wc mean   : ", sum(g[inI,inJ] .* logC_mu_wc * drdz));
println("    riskfree rate   : ", sum(g[inI,inJ] .* rf * drdz));
println("=============================================================")



#==============================================================================#
# Impulse Response Functions and Term structure of uncertainty prices
#==============================================================================#

# Which shock and how big
dW0 = Matrix(1. * I, 3, 3);

# Deciles under the baseline stationary distribution with robust decisions
gz_cdf = vec(cumsum(sum(g, dims=1)'* dz*dr, dims=1))
gr_cdf = vec(cumsum(sum(g, dims=2) * dz*dr, dims=1))
ind_r1dec = findfirst(x -> x >= 0.1, gr_cdf)
ind_r5dec = findfirst(x -> x >= 0.5, gr_cdf)
ind_r9dec = findfirst(x -> x >= 0.9, gr_cdf)

ind_z1dec = findfirst(x -> x >= 0.1, gz_cdf)
ind_z5dec = findfirst(x -> x >= 0.5, gz_cdf)
ind_z9dec = findfirst(x -> x >= 0.9, gz_cdf)

start_p = [[ind_r5dec, ind_z5dec],
           [ind_r1dec, ind_z5dec],
           [ind_r9dec, ind_z5dec],
           [ind_r5dec, ind_z1dec],
           [ind_r5dec, ind_z9dec]]

# horizon
hor = 1000
N = size(start_p)[1]                    # number of initial states

pii_irf = zeros(hor, 3, 2, N);
z_irf = zeros(hor, 3, 2, N);
price_12 = zeros(N, hor);
price_z = zeros(N, hor);


if compute_irfs == 1

    println(" (6) Compute Impulse Resonse Functions (hor=", hor, ")       ")
    d1_interp = LinearInterpolation((rr[:, 1], zz[1, :]), d1);
    d2_interp = LinearInterpolation((rr[:, 1], zz[1, :]), d2);
    P = factorize(sparse(1I, II*JJ, II*JJ) - A');
    P_wc = factorize(sparse(1I, II*JJ, II*JJ) - A_wc');
    P_noR = factorize(sparse(1I, II*JJ, II*JJ) - A_noR');

    hor_p = Int64[hor for i=1:N]
    dW0_p = Matrix{Float64}[dW0 for i=1:N]
    P_p = SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}[P for i=1:N]
    P_wc_p = SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}[P_wc for i=1:N]
    P_noR_p = SuiteSparse.UMFPACK.UmfpackLU{Float64,Int64}[P_noR for i=1:N]
    pii_p = Matrix{Float64}[pii for i=1:N]
    zz_p = Matrix{Float64}[zz for i=1:N]
    d1_interp_p = interpol[d1_interp for i=1:N]
    d2_interp_p = interpol[d2_interp for i=1:N]
    model_p = TwoCapitalEconomy[model for i=1:N]
    policies_p = PolicyFunctions[policies for i=1:N]
    grid_p = Grid_rz[grid for i=1:N]

    label_p = [1, 2, 3, 4, 5]

    res = pmap(IRF, start_p, dW0_p, hor_p, P_p, P_wc_p, P_noR_p,
                    pii_p, zz_p, d1_interp_p, d2_interp_p,
                    model_p, policies_p, grid_p, label_p)

    label_vec = [res[i][5] for i=1:N]

    for (i, spec) in enumerate(label_vec)
        pii_irf[:, :, :, spec] = res[i][1]
        z_irf[:, :, :, spec] = res[i][2]
        price_12[spec, :] = res[i][3]
        price_z[spec, :] = res[i][4]
    end
    println("=============================================================")
end





results = Dict("delta" => delta,
# Single capital
"alpha_c_hat" => alpha_c_hat, "beta_hat" => beta_hat,
"alpha_z_hat" => alpha_z_hat, "kappa_hat" => kappa_hat,
"sigma_c" => sigma_c, "sigma_z_1cap" => sigma_z_1cap,
"zbar" => zbar, "cons_1cap" => cons_1cap, "stdev_z_1cap" => stdev_z_1cap,
"H0" => H0, "H1" => H1,
# Two capital stocks
"alpha_k1_hat" => alpha_k1_hat, "alpha_k2_hat" => alpha_k2_hat,
"beta1_hat" => beta1_hat, "beta2_hat" => beta2_hat,
"sigma_k1" => sigma_k1, "sigma_k2" => sigma_k2,
"sigma_z" =>  sigma_z, "A1" => A1, "A2" => A2, "phi1" => phi1, "phi2" => phi2,
"alpha_z_tilde" => alpha_z_tilde, "kappa_tilde" => kappa_tilde,
"alpha_k1_tilde" => alpha_k1_tilde, "beta1_tilde" => beta1_tilde,
"alpha_k2_tilde" => alpha_k2_tilde, "beta2_tilde" => beta2_tilde,
"xi0" => xi0, "xi1" => xi1, "xi2" => xi2,
"I" => II, "J" => JJ,
"rmax" => rmax, "rmin" => rmin, "zmax" => zmax, "zmin" => zmin,
"rr" => rr, "zz" => zz, "pii" => pii, "dr" => dr, "dz" => dz, "T" => hor,
"maxit" => maxit, "crit" => crit, "Delta" => Delta, "inner" => inner,
# Without robustness
"V_noR" => V_noR, "val_noR" => val_noR,
"d1_F_noR" => d1_F_noR, "d2_F_noR" => d2_F_noR,
"d1_B_noR" => d1_B_noR, "d2_B_noR" => d2_B_noR,
"d1_noR" => d1_noR, "d2_noR" => d2_noR,
"g_noR_dist" => g_noR_dist, "g_noR" => g_noR,
"mu_1_noR" => mu_1_noR, "mu_r_noR" => mu_r_noR, "mu_z_noR" => mu_z_noR,
# Robust control under baseline
"V0" => V0, "V" => V, "val" => val, "ell_star" => ell_star,
"d1_F" => d1_F, "d2_F" => d2_F,
"d1_B" => d1_B, "d2_B" => d2_B,
"d1" => d1, "d2" => d2,
"h1_F" => policies.h1_F, "h2_F" => policies.h2_F, "hz_F" => policies.hz_F,
"h1_B" => policies.h1_B, "h2_B" => policies.h2_B, "hz_B" => policies.hz_B,
"h1_dist" => h1_dist, "h2_dist" => h2_dist, "hz_dist" => hz_dist,
"h1" => h1, "h2" => h2, "hz" => hz,
"g_dist" => g_dist, "g" => g,
"mu_1" => mu_1, "mu_r" => mu_r, "mu_z" => mu_z,
# Robust control under worst-case
"g_wc_dist" => g_wc_dist, "g_wc" => g_wc,
"mu_1_wc" => mu_1_wc, "mu_r_wc" => mu_r_wc, "mu_z_wc" => mu_z_wc,
# Non-robust control under worst-case
"g_wc_noR_dist" => g_wc_noR_dist, "g_wc_noR" => g_wc_noR,
# Distortion measures
"re" => re, "q" => q,
"chernoff" => chernoff, "halflife" => halflife,
# Local uncertainty prices (stationary distributions)
"h12_vec" => h12_vec, "h12_density" => h12_density,
"hz_vec" => hz_vec, "hz_density" => hz_density,
# Risk-free rate (stationary distributions)
"riskfree" => riskfree,
"rf_vec" => rf_vec, "rf_density" => rf_density,
# Consumption (stationary distributions)
"cons_noR" => cons_noR, "cons" => cons,
"cons_noR_vec" => cons_noR_vec, "cons_noR_density" => cons_noR_density,
"cons_vec" => cons_vec, "cons_density" => cons_density,
"cons_wc_vec" => cons_wc_vec, "cons_wc_density" => cons_wc_density,
# Consumption (drift and volatilities)
"logC_mu_noR" => logC_mu_noR, "logC_sigma_noR" => logC_sigma_noR,
"logC_mu" => logC_mu, "logC_sigma" => logC_sigma,
"logC_mu_wc" => logC_mu_wc, "logC_sigma_wc" => logC_sigma_wc,
# Impulse Response Functions
"R_irf" => pii_irf, "Z_irf" => z_irf,
# Expected future uncertainty prices
"shock_price_12" => price_12, "shock_price_z" => price_z,
# Calibration
"A_1cap" => A_1cap, "phi_1cap" => phi_1cap, "alpha_k_hat" => alpha_k_hat,
"consumption_investment" => consumption_investment, "investment_capital" => investment_capital)

npzwrite("./result_files/28/" * filename, results)
