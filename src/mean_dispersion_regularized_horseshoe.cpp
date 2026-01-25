#include "includes/mean_dispersion_regularized_horseshoe.h"
#include "includes/utils.h"
#include <cmath>
#include <vector>
#include <algorithm>

// ========================================
// Compute Regularized Shrinkage
// ========================================

// Regularized local shrinkage: λ̃² = (c² × λ²) / (c² + τ² × λ²)
double regularized_lambda_sq(double lambda_sq, double tau_sq, double c_sq) {
    return (c_sq * lambda_sq) / (c_sq + tau_sq * lambda_sq);
}

// Regularized global shrinkage: τ̃² = (c² × τ²) / (c² + τ₀² × τ²)
double regularized_tau_sq(double tau_sq, double tau_0_sq, double c_sq) {
    return (c_sq * tau_sq) / (c_sq + tau_0_sq * tau_sq);
}

// ========================================
// Initialize Regularized Horseshoe Parameters
// ========================================

RegularizedHorseshoeParams initialize_regularized_horseshoe_params(int J, int G, double p_0) {
    RegularizedHorseshoeParams params;

    // Initialize lambda (local shrinkage) to 1.0
    params.lambda.resize(J);
    params.nu.resize(J);
    for (int j = 0; j < J; j++) {
        params.lambda[j] = Eigen::VectorXd::Ones(G);
        params.nu[j] = Eigen::VectorXd::Ones(G);
    }

    // Initialize tau (global shrinkage per cluster)
    params.tau.resize(J, 1.0);
    params.xi.resize(J, 1.0);

    // Initialize overall global scale
    params.tau_0 = 1.0;
    params.xi_tau_0 = 1.0;

    // Initialize overall variance
    params.sigma_mu = 1.0;

    // Initialize regularization parameter (slab variance)
    // Following Piironen & Vehtari: c² = (p_0 / (D - p_0)) × (σ² / n)
    // For our case: c² set based on expected number of differential genes
    params.p_0 = p_0;  // Expected number of non-zero effects per cluster

    // Set c² based on sparsity assumption
    // If p_0 genes are differential out of G total:
    double sparsity_ratio = p_0 / G;
    params.c_squared = sparsity_ratio / (1.0 - sparsity_ratio);

    // Ensure reasonable bounds
    params.c_squared = std::max(0.1, std::min(10.0, params.c_squared));

    return params;
}

// ========================================
// Update tau_0 (overall global hyperprior)
// ========================================

void update_tau_0_regularized(RegularizedHorseshoeParams& rhs_params, int J) {
    // tau_0^2 ~ InvGamma(J/2, 1/xi_tau_0 + sum(1/tau²_j) / 2)
    // Note: We use unregularized tau in the prior for tau_0

    double sum_inv_tau_sq = 0.0;
    for (int j = 0; j < J; j++) {
        double tau_sq = rhs_params.tau[j] * rhs_params.tau[j];
        if (tau_sq > 1e-10) {
            sum_inv_tau_sq += 1.0 / tau_sq;
        }
    }

    double shape = J / 2.0;
    double rate = 1.0 / rhs_params.xi_tau_0 + sum_inv_tau_sq / 2.0;

    // Safeguard
    if (rate < 1e-6) rate = 1e-6;
    if (rate > 1e6) rate = 1e6;

    std::gamma_distribution<double> gamma_dist(shape, 1.0 / rate);
    double tau_0_sq_inv = gamma_dist(rng_local);
    rhs_params.tau_0 = 1.0 / std::sqrt(tau_0_sq_inv);

    // Constrain tau_0 to reasonable range
    rhs_params.tau_0 = std::max(0.1, std::min(5.0, rhs_params.tau_0));

    // Update auxiliary variable xi_tau_0
    double rate_xi = 1.0 + 1.0 / (rhs_params.tau_0 * rhs_params.tau_0);

    std::gamma_distribution<double> gamma_dist_xi(1.0, 1.0 / rate_xi);
    double xi_inv = gamma_dist_xi(rng_local);
    rhs_params.xi_tau_0 = 1.0 / xi_inv;
}

// ========================================
// Update tau_j (cluster-level global shrinkage)
// ========================================

void update_tau_regularized(
    RegularizedHorseshoeParams& rhs_params,
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::VectorXd& mu_baseline,
    int J, int G
) {
    for (int j = 0; j < J; j++) {
        double sum_term = 0.0;
        int valid_genes = 0;

        for (int g = 0; g < G; g++) {
            double mu_jg = mu_star_1_J(j, g);
            double mu_bg = mu_baseline(g);

            if (mu_jg <= 0 || mu_bg <= 0 || !std::isfinite(mu_jg) || !std::isfinite(mu_bg)) {
                continue;
            }

            double delta_jg = std::log(mu_jg) - std::log(mu_bg);
            if (!std::isfinite(delta_jg) || std::abs(delta_jg) > 10.0) {
                continue;
            }

            double lambda_sq = rhs_params.lambda[j](g) * rhs_params.lambda[j](g);
            double sigma_mu_sq = rhs_params.sigma_mu * rhs_params.sigma_mu;
            double tau_sq = rhs_params.tau[j] * rhs_params.tau[j];

            // Use REGULARIZED lambda
            double lambda_tilde_sq = regularized_lambda_sq(lambda_sq, tau_sq, rhs_params.c_squared);

            sum_term += (delta_jg * delta_jg) / (lambda_tilde_sq * sigma_mu_sq);
            valid_genes++;
        }

        if (valid_genes < 10) continue;

        // Normalize by number of genes to prevent accumulation
        double mean_term = sum_term / valid_genes;

        double tau_0_sq = rhs_params.tau_0 * rhs_params.tau_0;
        double xi_j = rhs_params.xi[j];

        // tau_j^2 ~ InvGamma(1, 1/(xi_j × τ₀²) + mean_term/2)
        double shape_tau = 1.0;
        double rate_tau = 1.0 / (xi_j * tau_0_sq) + mean_term / 2.0;

        if (rate_tau < 1e-6) rate_tau = 1e-6;
        if (rate_tau > 1e6) rate_tau = 1e6;

        std::gamma_distribution<double> gamma_dist_tau(shape_tau, 1.0 / rate_tau);
        double tau_sq_inv = gamma_dist_tau(rng_local);
        double tau_new = 1.0 / std::sqrt(tau_sq_inv);

        // Constrain tau
        const double MAX_TAU = 5.0;
        const double MIN_TAU = 0.01;
        if (tau_new > MAX_TAU) tau_new = MAX_TAU;
        if (tau_new < MIN_TAU) tau_new = MIN_TAU;

        rhs_params.tau[j] = tau_new;

        // Update xi_j
        double tau_j_sq = rhs_params.tau[j] * rhs_params.tau[j];
        double rate_xi = 1.0 + 1.0 / (tau_j_sq * tau_0_sq);

        std::gamma_distribution<double> gamma_dist_xi(1.0, 1.0 / rate_xi);
        rhs_params.xi[j] = 1.0 / gamma_dist_xi(rng_local);
    }
}

// ========================================
// Update c² (slab variance) - KEY REGULARIZATION PARAMETER
// ========================================

void update_c_squared(
    RegularizedHorseshoeParams& rhs_params,
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::VectorXd& mu_baseline,
    int J, int G
) {
    // Following Piironen & Vehtari:
    // c² can be learned from data or fixed based on prior information

    // Option 1: Keep fixed based on sparsity assumption (conservative)
    // rhs_params.c_squared stays as initialized

    // Option 2: Update based on empirical sparsity (adaptive)
    // Count effective number of "large" deviations
    int count_large = 0;

    for (int j = 0; j < J; j++) {
        for (int g = 0; g < G; g++) {
            double mu_jg = mu_star_1_J(j, g);
            double mu_bg = mu_baseline(g);

            if (mu_jg <= 0 || mu_bg <= 0) continue;

            double delta = std::log(mu_jg) - std::log(mu_bg);
            if (!std::isfinite(delta)) continue;

            // Consider "large" if |delta| > 0.5 (1.6x fold-change)
            if (std::abs(delta) > 0.5) {
                count_large++;
            }
        }
    }

    // Update p_0 (effective number of non-zeros)
    double p_eff = static_cast<double>(count_large) / J;  // Average per cluster
    rhs_params.p_0 = 0.9 * rhs_params.p_0 + 0.1 * p_eff;  // Smooth update

    // Update c² based on updated p_0
    double sparsity_ratio = rhs_params.p_0 / G;
    sparsity_ratio = std::max(0.01, std::min(0.99, sparsity_ratio));

    rhs_params.c_squared = sparsity_ratio / (1.0 - sparsity_ratio);

    // Constrain to reasonable range
    rhs_params.c_squared = std::max(0.1, std::min(10.0, rhs_params.c_squared));
}

// ========================================
// Main MCMC Function
// ========================================

MeanDispersionRegularizedHorseshoeResult mean_dispersion_regularized_horseshoe_mcmc(
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    const Eigen::VectorXd& mu_baseline,
    RegularizedHorseshoeParams& rhs_params,
    double v_1,
    double v_2,
    const Eigen::VectorXd& m_b,
    bool quadratic
) {
    int J = mu_star_1_J.rows();
    int G = mu_star_1_J.cols();

    MeanDispersionRegularizedHorseshoeResult result;

    // ============================================================
    // 1. Update local shrinkage parameters (lambda_jg, nu_jg)
    // ============================================================

    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            double mu_jg = mu_star_1_J(j, g);
            double mu_bg = mu_baseline(g);

            if (mu_jg <= 0 || mu_bg <= 0 || !std::isfinite(mu_jg) || !std::isfinite(mu_bg)) {
                continue;
            }

            double delta_jg = std::log(mu_jg) - std::log(mu_bg);
            if (!std::isfinite(delta_jg) || std::abs(delta_jg) > 10.0) continue;

            double tau_sq = rhs_params.tau[j] * rhs_params.tau[j];
            double sigma_mu_sq = rhs_params.sigma_mu * rhs_params.sigma_mu;
            double nu_jg = rhs_params.nu[j](g);

            // Update lambda_jg (standard horseshoe update, regularization applied in likelihood)
            // λ²_jg ~ InvGamma(1, 1/ν_jg + δ²/(2 × τ² × σ²))
            double rate_lambda = 1.0 / nu_jg +
                                (delta_jg * delta_jg) / (2.0 * tau_sq * sigma_mu_sq);

            if (rate_lambda < 1e-6) rate_lambda = 1e-6;
            if (rate_lambda > 1e6) rate_lambda = 1e6;

            std::gamma_distribution<double> gamma_dist_lambda(1.0, 1.0 / rate_lambda);
            double lambda_sq_inv = gamma_dist_lambda(rng_local);
            rhs_params.lambda[j](g) = 1.0 / std::sqrt(lambda_sq_inv);

            // Constrain lambda
            rhs_params.lambda[j](g) = std::max(0.01, std::min(100.0, rhs_params.lambda[j](g)));

            // Update nu_jg
            double lambda_jg = rhs_params.lambda[j](g);
            double rate_nu = 1.0 + 1.0 / (lambda_jg * lambda_jg);

            std::gamma_distribution<double> gamma_dist_nu(1.0, 1.0 / rate_nu);
            double nu_inv = gamma_dist_nu(rng_local);
            rhs_params.nu[j](g) = 1.0 / nu_inv;
        }
    }

    // ============================================================
    // 2. Update cluster-level global shrinkage (tau_j, xi_j)
    // ============================================================
    update_tau_regularized(rhs_params, mu_star_1_J, mu_baseline, J, G);

    // ============================================================
    // 3. Update overall global scale (tau_0, xi_tau_0)
    // ============================================================
    update_tau_0_regularized(rhs_params, J);

    // ============================================================
    // 4. Update regularization parameter (c²)
    // ============================================================
    update_c_squared(rhs_params, mu_star_1_J, mu_baseline, J, G);

    // ============================================================
    // 5. Update global variance (sigma_mu)
    // ============================================================

    double sum_all = 0.0;
    int valid_count = 0;

    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            double mu_jg = mu_star_1_J(j, g);
            double mu_bg = mu_baseline(g);

            if (mu_jg <= 0 || mu_bg <= 0) continue;

            double delta_jg = std::log(mu_jg) - std::log(mu_bg);
            if (!std::isfinite(delta_jg)) continue;

            double tau_sq = rhs_params.tau[j] * rhs_params.tau[j];
            double lambda_sq = rhs_params.lambda[j](g) * rhs_params.lambda[j](g);

            // Use REGULARIZED shrinkage
            double lambda_tilde_sq = regularized_lambda_sq(lambda_sq, tau_sq, rhs_params.c_squared);
            double tau_0_sq = rhs_params.tau_0 * rhs_params.tau_0;
            double tau_tilde_sq = regularized_tau_sq(tau_sq, tau_0_sq, rhs_params.c_squared);

            sum_all += (delta_jg * delta_jg) / (tau_tilde_sq * lambda_tilde_sq);
            valid_count++;
        }
    }

    if (valid_count > 0) {
        // Normalize to prevent accumulation
        double mean_term = sum_all / valid_count;

        double a_sigma = 2.0;
        double b_sigma = 1.0;

        double shape_sigma = a_sigma + valid_count / 2.0;
        double rate_sigma = b_sigma + mean_term * valid_count / 2.0;

        std::gamma_distribution<double> gamma_dist_sigma(shape_sigma, 1.0 / rate_sigma);
        double sigma_mu_sq_inv = gamma_dist_sigma(rng_local);
        rhs_params.sigma_mu = 1.0 / std::sqrt(sigma_mu_sq_inv);

        // Constrain
        rhs_params.sigma_mu = std::max(0.1, std::min(10.0, rhs_params.sigma_mu));
    }

    // ============================================================
    // 6. Update dispersion regression parameters (unchanged)
    // ============================================================

    std::vector<double> x_vals, y_vals;

    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            double mu = mu_star_1_J(j, g);
            double phi = phi_star_1_J(j, g);

            if (std::isfinite(std::log(mu)) && std::isfinite(std::log(phi)) &&
                mu > 0 && phi > 0) {
                x_vals.push_back(std::log(mu));
                y_vals.push_back(std::log(phi));
            }
        }
    }

    int n = x_vals.size();

    if (n == 0) {
        result.alpha_phi_2 = 1.0;
        result.b = m_b;
        result.rhs_params = rhs_params;
        result.log_likelihood = -std::numeric_limits<double>::infinity();
        return result;
    }

    Eigen::VectorXd y(n);
    Eigen::MatrixXd X;

    for (int i = 0; i < n; ++i) {
        y(i) = y_vals[i];
    }

    if (quadratic) {
        X.resize(n, 3);
        for (int i = 0; i < n; ++i) {
            X(i, 0) = 1.0;
            X(i, 1) = x_vals[i];
            X(i, 2) = x_vals[i] * x_vals[i];
        }
    } else {
        X.resize(n, 2);
        for (int i = 0; i < n; ++i) {
            X(i, 0) = 1.0;
            X(i, 1) = x_vals[i];
        }
    }

    Eigen::VectorXd b_hat = (X.transpose() * X).ldlt().solve(X.transpose() * y);

    double shape = v_1 / 2.0;
    double scale = v_2 / 2.0;

    std::gamma_distribution<double> gamma_dist(shape, 1.0 / scale);
    double gamma_sample = gamma_dist(rng_local);
    double alpha_phi_2 = 1.0 / gamma_sample;

    Eigen::MatrixXd cov_b = alpha_phi_2 * (X.transpose() * X).inverse();
    Eigen::VectorXd b_new = rmvnorm(b_hat, cov_b);

    result.alpha_phi_2 = alpha_phi_2;
    result.b = b_new;
    result.rhs_params = rhs_params;

    // ============================================================
    // 7. Calculate Log-Likelihood
    // ============================================================

    double log_lik_delta = 0.0;
    double const_part = -0.5 * std::log(2.0 * M_PI);

    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            double mu_jg = mu_star_1_J(j, g);
            double mu_bg = mu_baseline(g);

            if (mu_jg <= 0 || mu_bg <= 0) continue;

            double delta = std::log(mu_jg) - std::log(mu_bg);
            if (!std::isfinite(delta)) continue;

            double tau_sq = rhs_params.tau[j] * rhs_params.tau[j];
            double lambda_sq = rhs_params.lambda[j](g) * rhs_params.lambda[j](g);
            double sigma_sq_mu = rhs_params.sigma_mu * rhs_params.sigma_mu;

            // Use REGULARIZED variances
            double lambda_tilde_sq = regularized_lambda_sq(lambda_sq, tau_sq, rhs_params.c_squared);
            double tau_0_sq = rhs_params.tau_0 * rhs_params.tau_0;
            double tau_tilde_sq = regularized_tau_sq(tau_sq, tau_0_sq, rhs_params.c_squared);

            double total_var = sigma_sq_mu * tau_tilde_sq * lambda_tilde_sq;

            log_lik_delta += const_part - 0.5 * std::log(total_var)
                           - (delta * delta) / (2.0 * total_var);
        }
    }

    double log_lik_phi = 0.0;
    if (n > 0) {
        double rss = (y - X * b_new).squaredNorm();
        log_lik_phi = n * const_part - 0.5 * n * std::log(alpha_phi_2)
                     - rss / (2.0 * alpha_phi_2);
    }

    result.log_likelihood = log_lik_delta + log_lik_phi;

    return result;
}