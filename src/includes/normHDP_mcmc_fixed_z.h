#ifndef NORMHDP_MCMC_FIXED_Z_H
#define NORMHDP_MCMC_FIXED_Z_H

#include <RcppEigen.h>
#include <vector>
#include <Eigen/Dense>

// Declaration must match the .cpp definition types exactly
Rcpp::List normHDP_mcmc_fixed_z(
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<std::vector<int>>& cluster_estimates,
    int number_iter,
    int thinning,
    bool empirical,
    int burn_in,
    bool quadratic,
    int iter_update,
    double beta_mean,
    double alpha_mu_2,
    double adaptive_prop,
    int num_cores,
    Rcpp::NumericVector baynorm_mu_estimate,
    Rcpp::NumericVector baynorm_phi_estimate,
    Rcpp::List baynorm_beta
);

#endif // NORMHDP_MCMC_FIXED_Z_H