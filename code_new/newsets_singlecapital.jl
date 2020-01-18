#==============================================================================#
#  Structured Unceratinty and its price: single capital example
#
#  Author: Balint Szoke
#  Date: Nov 2017
#==============================================================================#
#===============         Single capital economy        ========================#
#==============================================================================#

using Pkg
# Pkg.add("Optim")
# Pkg.add("Roots")
# Pkg.add("Interpolations")
using Optim
using Roots
using NPZ

kappa_tilde = parse(Float64, ARGS[1])
filename = ARGS[2]

include("newsets_utils.jl")
#==============================================================================#
#    PARAMETERS
#==============================================================================#
delta = .002;

# (1) Baseline model
alpha_z_hat = .0;
kappa_hat = .014;

alpha_c_hat = .484;       # consumption intercept (estimated)
beta_hat = 1.;

sigma_c = [.477, .0  ];   # consumption exposure (= exposure of single capital)
sigma_z = [.011, .025];

#===========================  CALIBRATION  ====================================#
# consumption_investment = 3.1;
# A, phi, alpha_k_hat, investment_capital = calibration2(40.,
#                                             consumption_investment,
#                                             alpha_c_hat, delta, sigma_c)


A = .05
phi = 28.
investment_capital, consumption_investment, alpha_k_hat = calibration3(phi,
                                            A, delta, alpha_c_hat, sigma_c)

println("  Calibrated values: A:", A,
        "  phi: ", phi,
        "  alpha_k : ", alpha_k_hat,
        "  C/I : ", consumption_investment,
        "  I/K : ", investment_capital)
println("=============================================================")
#==============================================================================#
baseline = Baseline(alpha_z_hat, kappa_hat, sigma_z,
                    alpha_c_hat, beta_hat, sigma_c, delta);
technology = Technology(A, phi);

# optimal investment-capital ratio:
d_star = dstar_singlecapital(technology, baseline);

# single capital case should give rise to observed consumption growth:
alpha_c2 = 100*(d_star-phi*d_star^2/2)+alpha_k_hat-(.01)*dot(sigma_c,sigma_c)/2;

println("alpha_c_hat =", baseline.alpha_k_hat,
        " technology parameters imply ", alpha_c2, '\n');


#=================== Table 1&2 ================================================#
kappa_list = [.01, .005];
kappa_list2 = [.014, .01, kappa_tilde];
q_list = [.1, .2];

store_matrix = zeros(length(kappa_list)+length(kappa_list2)*length(q_list), 25);

for ki=1:length(kappa_list)
    H0, H1, ak, b, az, k, dc, re, q, l_star = worst_case(alpha_z_hat,
                                                            kappa_list[ki],
                                                            baseline.alpha_k_hat,
                                                            baseline.beta_hat,
                                                            baseline, technology);
    store_matrix[ki, 1:4] = [q, alpha_z_hat, kappa_list[ki], baseline.beta_hat]
    store_matrix[ki, 5:11] = [ak, b, az, k, dc, re, q];
    res = optimize(r -> chernoff_objective_single(r, H0, H1, baseline), 0., 1.);
    chernoff = - Optim.minimum(res)
    if chernoff == 0.
        halflife = Inf
    else
        halflife = log(2.) / chernoff
    end
    store_matrix[ki, 12:13] = [chernoff, halflife];
    q, re, c, hl = worrisome_entropy(alpha_z_hat, kappa_list[ki],
                                        baseline.alpha_k_hat, baseline.beta_hat, baseline);
    store_matrix[ki, 14:17] = [q, re, c, hl]
    store_matrix[ki, 18:19] = H0'
    store_matrix[ki, 20:21] = H1'
    store_matrix[ki, 22] = l_star
    x0, x1, x2 = tilting_function(Worrisome(alpha_z_hat,
                                            kappa_list[ki],
                                            baseline.alpha_k_hat,
                                            baseline.beta_hat), baseline)
    store_matrix[ki, 23:25] = [x0, x1, x2]
end

for ki=1:length(kappa_list2)
    for qi=1:length(q_list)
        alpha_tilde, wc = choose_alpha(q_list[qi],kappa_list2[ki],
                                baseline.alpha_k_hat,baseline.beta_hat,baseline,technology);
        H0, H1, ak, b, az, k, dc, re, q, l_star = wc;

        row_index = length(kappa_list) + (ki-1)*length(q_list) + qi
        store_matrix[row_index, 1:4] = [q, alpha_tilde, kappa_list2[ki], baseline.beta_hat];
        store_matrix[row_index, 5:11] = [ak, b, az, k, dc, re, q];
        res = optimize(r -> chernoff_objective_single(r,H0,H1,baseline), 0., 1.);
        chernoff = - Optim.minimum(res)
        if chernoff == 0.
            halflife = Inf
        else
            halflife = log(2.) / chernoff
        end
        store_matrix[row_index, 12:13] = [chernoff, halflife];
        q, re, c, hl = worrisome_entropy(alpha_tilde, kappa_list2[ki],
                                        baseline.alpha_k_hat, baseline.beta_hat, baseline);
        store_matrix[row_index, 14:17] = [q, re, c, hl]
        store_matrix[row_index, 18:19] = H0'
        store_matrix[row_index, 20:21] = H1'
        store_matrix[row_index, 22] = l_star
        x0, x1, x2 = tilting_function(Worrisome(alpha_tilde,
                                                kappa_list2[ki],
                                                baseline.alpha_k_hat,
                                                baseline.beta_hat), baseline)
        store_matrix[row_index, 23:25] = [x0, x1, x2]
    end
end

store_matrix2 = zeros(length(kappa_list2[2:end])*length(q_list), 25);

for ki=1:length(kappa_list2[2:end])
    for qi=1:length(q_list)
        row_index = (ki-1)*length(q_list) + qi

        # alpha_z_tilde = minimum_xi(store_matrix[length(kappa_list) + row_index , 2],
        #                            alpha_z_hat, sigma_z)
        alpha_z_tilde = store_matrix[length(kappa_list) + 2 + row_index , 2]
        beta_tilde, wc = choose_beta(q_list[qi], alpha_z_tilde, baseline.kappa_hat,
                                    baseline.alpha_k_hat, baseline, technology);
        H0, H1, ak, b, az, k, dc, re, q, l_star = wc;

        store_matrix2[row_index, 1:4] = [q, alpha_z_tilde, baseline.kappa_hat, beta_tilde];
        store_matrix2[row_index, 5:11] = [ak, b, az, k, dc, re, q];
        res = optimize(r -> chernoff_objective_single(r,H0,H1,baseline), 0., 1.);
        chernoff = - Optim.minimum(res)
        if chernoff == 0.
            halflife = Inf
        else
            halflife = log(2.) / chernoff
        end
        store_matrix2[row_index, 12:13] = [chernoff, halflife];
        q, re, c, hl = worrisome_entropy(alpha_z_tilde, baseline.kappa_hat,
                                        baseline.alpha_k_hat, beta_tilde, baseline);
        store_matrix2[row_index, 14:17] = [q, re, c, hl]
        store_matrix2[row_index, 18:19] = H0'
        store_matrix2[row_index, 20:21] = H1'
        store_matrix2[row_index, 22] = l_star
        x0, x1, x2 = tilting_function(Worrisome(alpha_z_tilde,
                                                baseline.kappa_hat,
                                                baseline.alpha_k_hat,
                                                beta_tilde), baseline)
        store_matrix2[row_index, 23:25] = [x0, x1, x2]

    end
end
# npzwrite("./data/model_singlecapital.npz", vcat(store_matrix, store_matrix2))
# res = Dict("res" => vcat(store_matrix, store_matrix2), 
#     "A" => A,
#     "phi" => phi,
#     "alpha_k" => alpha_k_hat,
#     "c" => consumption_investment,
#     "i" => investment_capital,
#     "alpha_c_hat" => baseline.alpha_k_hat,
#     "tech" => alpha_c2,
#     "sigma_z" => sigma_z)
npzwrite("../data/" * filename, vcat(store_matrix, store_matrix2))

