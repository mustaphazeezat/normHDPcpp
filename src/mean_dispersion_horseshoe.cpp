#include "includes/mean_dispersion_horseshoe.h"
#include "includes/utils.h"
#include <cmath>
#include <vector>

// ========================================
// Update tau_j using Makalic-Schmidt approach
// Removes redundant tau_0 layer for better identifiability
// ========================================

void update_tau_makalic_schmidt(
    HorseshoeParams& horseshoe_params,
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::VectorXd& mu_baseline,
    int J, int G
) {
    for (int j = 0; j < J; j++) {
        // Compute sum of normalized squared deviations
        double sum_term = 0.0;
        for (int g = 0; g < G; g++) {
            double delta_jg = std::log(mu_star_1_J(j, g)) - std::log(mu_baseline(g));
            double lambda_jg = horseshoe_params.lambda[j](g);
            double sigma_mu = horseshoe_params.sigma_mu;

            // Contribution to likelihood
            sum_term += (delta_jg * delta_jg) / (lambda_jg * lambda_jg * sigma_mu * sigma_mu);
        }

        // Update tau_j using Half-Cauchy(0, 1) prior
        // Implemented via auxiliary variable xi_j

        // tau_j^2 ~ InvGamma(G/2, 1/xi_j + sum_term/2)
        double shape_tau = G / 2.0;
        double rate_tau = 1.0 / horseshoe_params.xi[j] + sum_term / 2.0;

        std::gamma_distribution<double> gamma_dist_tau(shape_tau, 1.0 / rate_tau);
        double tau_sq_inv = gamma_dist_tau(rng_local);
        horseshoe_params.tau[j] = 1.0 / std::sqrt(tau_sq_inv);

        // Update auxiliary variable xi_j for Half-Cauchy(0, 1)
        // xi_j ~ InvGamma(1, 1 + 1/tau_j^2)
        double tau_j_sq = horseshoe_params.tau[j] * horseshoe_params.tau[j];
        double rate_xi = 1.0 + 1.0 / tau_j_sq;

        std::gamma_distribution<double> gamma_dist_xi(1.0, 1.0 / rate_xi);
        double xi_inv = gamma_dist_xi(rng_local);
        horseshoe_params.xi[j] = 1.0 / xi_inv;
    }
}

MeanDispersionHorseshoeResult mean_dispersion_horseshoe_mcmc(
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    const Eigen::VectorXd& mu_baseline,
    HorseshoeParams& horseshoe_params,
    double v_1,
    double v_2,
    const Eigen::VectorXd& m_b,
    bool quadratic
) {
    int J = mu_star_1_J.rows();
    int G = mu_star_1_J.cols();

    MeanDispersionHorseshoeResult result;

    // ============================================================
    // 1. Update local shrinkage parameters (lambda_jg, nu_jg)
    // ============================================================

    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            // Deviation from baseline
            double delta_jg = std::log(mu_star_1_J(j, g)) - std::log(mu_baseline(g));

            // Update lambda_jg (local shrinkage)
            // lambda_jg^2 ~ InvGamma(1, 1/nu_jg + delta_jg^2/(2*tau_j^2*sigma_mu^2))
            double tau_j = horseshoe_params.tau[j];
            double sigma_mu = horseshoe_params.sigma_mu;
            double nu_jg = horseshoe_params.nu[j](g);

            double rate_lambda = 1.0 / nu_jg +
                                (delta_jg * delta_jg) / (2.0 * tau_j * tau_j * sigma_mu * sigma_mu);

            // Sample lambda_jg^2 from InvGamma(1, rate)
            std::gamma_distribution<double> gamma_dist_lambda(1.0, 1.0 / rate_lambda);
            double lambda_sq_inv = gamma_dist_lambda(rng_local);
            horseshoe_params.lambda[j](g) = 1.0 / std::sqrt(lambda_sq_inv);

            // Update auxiliary variable nu_jg
            // nu_jg ~ InvGamma(1, 1 + 1/lambda_jg^2)
            double lambda_jg = horseshoe_params.lambda[j](g);
            double rate_nu = 1.0 + (1.0 / (lambda_jg * lambda_jg));

            std::gamma_distribution<double> gamma_dist_nu(1.0, 1.0 / rate_nu);
            double nu_inv = gamma_dist_nu(rng_local);
            horseshoe_params.nu[j](g) = 1.0 / nu_inv;
        }
    }

    // ============================================================
    // 2. Update global shrinkage parameters (tau_j, xi_j)
    //    Using Makalic-Schmidt approach (NO tau_0)
    // ============================================================
    update_tau_makalic_schmidt(horseshoe_params, mu_star_1_J, mu_baseline, J, G);

    // ============================================================
    // 3. Update global variance parameter (sigma_mu)
    // ============================================================

    double sum_all = 0.0;
    for (int j = 0; j < J; ++j) {
        for (int g = 0; g < G; ++g) {
            double delta_jg = std::log(mu_star_1_J(j, g)) - std::log(mu_baseline(g));
            double tau_j = horseshoe_params.tau[j];
            double lambda_jg = horseshoe_params.lambda[j](g);
            sum_all += (delta_jg * delta_jg) / (tau_j * tau_j * lambda_jg * lambda_jg);
        }
    }

    // sigma_mu^2 ~ InvGamma(a_sigma + J*G/2, b_sigma + sum_all/2)
    // Using weak prior to let data determine scale
    double a_sigma = 1.0;   // Weak prior shape
    double b_sigma = 0.01;  // Weak prior scale

    double shape_sigma = a_sigma + (J * G) / 2.0;
    double rate_sigma = b_sigma + sum_all / 2.0;

    std::gamma_distribution<double> gamma_dist_sigma(shape_sigma, 1.0 / rate_sigma);
    double sigma_mu_sq_inv = gamma_dist_sigma(rng_local);
    horseshoe_params.sigma_mu = 1.0 / std::sqrt(sigma_mu_sq_inv);

    // ============================================================
    // 4. Update dispersion regression parameters
    // ============================================================

    // Collect valid log values for regression
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
        // Return defaults if no valid data
        result.alpha_phi_2 = 1.0;
        result.b = m_b;
        result.horseshoe_params = horseshoe_params;
        result.log_likelihood = -std::numeric_limits<double>::infinity();
        return result;
    }

    // Fit linear regression for dispersion
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

    // Least squares: b = (X'X)^-1 X'y
    Eigen::VectorXd b_hat = (X.transpose() * X).ldlt().solve(X.transpose() * y);

    // Sample alpha_phi_2 from inverse gamma
    double shape = v_1 / 2.0;
    double scale = v_2 / 2.0;

    std::gamma_distribution<double> gamma_dist(shape, 1.0 / scale);
    double gamma_sample = gamma_dist(rng_local);
    double alpha_phi_2 = 1.0 / gamma_sample;

    // Sample b from multivariate normal
    Eigen::MatrixXd cov_b = alpha_phi_2 * (X.transpose() * X).inverse();
    Eigen::VectorXd b_new = rmvnorm(b_hat, cov_b);

    // ============================================================
    // 5. Return results
    // ============================================================

    result.alpha_phi_2 = alpha_phi_2;
    result.b = b_new;
    result.horseshoe_params = horseshoe_params;

    // ============================================================
    // 6. Calculate Log-Likelihood
    // ============================================================
    double log_lik_delta = 0.0;
    double const_part = -0.5 * std::log(2.0 * M_PI);

    for (int j = 0; j < J; ++j) {
        double tau_sq = horseshoe_params.tau[j] * horseshoe_params.tau[j];
        for (int g = 0; g < G; ++g) {
            double delta = std::log(mu_star_1_J(j, g)) - std::log(mu_baseline(g));
            double lambda_sq = horseshoe_params.lambda[j](g) * horseshoe_params.lambda[j](g);
            double sigma_sq_mu = horseshoe_params.sigma_mu * horseshoe_params.sigma_mu;

            // Total variance for this gene/cluster pair
            double total_var = sigma_sq_mu * tau_sq * lambda_sq;

            log_lik_delta += const_part - 0.5 * std::log(total_var) - (delta * delta) / (2.0 * total_var);
        }
    }

    // Log-likelihood of the dispersion residuals
    double log_lik_phi = 0.0;
    if (n > 0) {
        double rss = (y - X * b_new).squaredNorm();
        log_lik_phi = n * const_part - 0.5 * n * std::log(alpha_phi_2) - rss / (2.0 * alpha_phi_2);
    }

    result.log_likelihood = log_lik_delta + log_lik_phi;

    return result;
}

// Initialize horseshoe parameters
HorseshoeParams initialize_horseshoe_params(int J, int G) {
    HorseshoeParams params;

    // Initialize lambda (local shrinkage) to 1.0
    params.lambda.resize(J);
    params.nu.resize(J);
    for (int j = 0; j < J; j++) {
        params.lambda[j] = Eigen::VectorXd::Ones(G);
        params.nu[j] = Eigen::VectorXd::Ones(G);
    }

    // Initialize tau (global shrinkage per cluster) to 1.0 (neutral)
    params.tau.resize(J, 1.0);
    params.xi.resize(J, 1.0);

    // Initialize overall variance
    params.sigma_mu = 1.0;

    return params;
}