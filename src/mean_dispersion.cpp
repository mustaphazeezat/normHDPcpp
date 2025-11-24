#include "includes/mean_dispersion.h"
#include "includes/utils.h"
#include <cmath>
#include <vector>

MeanDispersionResult mean_dispersion_mcmc(
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    double v_1,
    double v_2,
    const Eigen::VectorXd& m_b,
    bool quadratic
) {
    int J = mu_star_1_J.rows();
    int G = mu_star_1_J.cols();
    
    // Collect valid log values
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
        MeanDispersionResult result;
        result.alpha_phi_2 = 1.0;
        result.b = m_b;
        return result;
    }
    
    // Fit linear regression
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
    
    // Residual sum of squares
    Eigen::VectorXd residuals = y - X * b_hat;
    double rss = residuals.squaredNorm();
    int df = n - b_hat.size();
    double rse_squared = (df > 0) ? (rss / df) : 1.0;
    
    // Sample alpha_phi_2 from inverse gamma
    double shape = v_1 / 2.0;
    double scale = v_2 / 2.0;
    
    // Sample from gamma, then take reciprocal
    std::gamma_distribution<double> gamma_dist(shape, 1.0 / scale);
    double gamma_sample = gamma_dist(rng_local);
    double alpha_phi_2 = 1.0 / gamma_sample;
    
    // Sample b from multivariate normal
    Eigen::MatrixXd cov_b = alpha_phi_2 * (X.transpose() * X).inverse();
    Eigen::VectorXd b_new = rmvnorm(b_hat, cov_b);
    
    MeanDispersionResult result;
    result.alpha_phi_2 = alpha_phi_2;
    result.b = b_new;
    
    return result;
}