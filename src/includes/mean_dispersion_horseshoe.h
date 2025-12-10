#ifndef MEAN_DISPERSION_HORSESHOE_H
#define MEAN_DISPERSION_HORSESHOE_H

#include <RcppEigen.h>
#include <vector>

// Horseshoe prior parameters structure
struct HorseshoeParams {
    std::vector<Eigen::VectorXd> lambda;  // J x G local shrinkage parameters
    std::vector<Eigen::VectorXd> nu;      // J x G auxiliary variables for lambda
    std::vector<double> tau;               // J global shrinkage parameters (one per cluster)
    std::vector<double> xi;                // J auxiliary variables for tau
    double sigma_mu;                       // Overall variance parameter
};

// Result structure
struct MeanDispersionHorseshoeResult {
    double alpha_phi_2;
    Eigen::VectorXd b;
    HorseshoeParams horseshoe_params;
};

// Main MCMC update function with horseshoe prior
MeanDispersionHorseshoeResult mean_dispersion_horseshoe_mcmc(
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    const Eigen::VectorXd& mu_baseline,
    HorseshoeParams& horseshoe_params,
    double v_1,
    double v_2,
    const Eigen::VectorXd& m_b,
    bool quadratic
);

// Initialize horseshoe parameters
HorseshoeParams initialize_horseshoe_params(int J, int G);

#endif // MEAN_DISPERSION_HORSESHOE_H