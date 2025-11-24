#ifndef ENSEMBLE_MCMC_H
#define ENSEMBLE_MCMC_H

#include <Eigen/Dense>
#include <vector>
#include "normHDP_mcmc.h"

// Structure for ensemble averaged results
struct EnsembleResult {
    // Averaged posterior estimates
    Eigen::VectorXd b_mean;
    Eigen::VectorXd b_sd;

    double alpha_phi2_mean;
    double alpha_phi2_sd;

    Eigen::MatrixXd P_J_D_mean;  // J x D
    Eigen::MatrixXd P_J_D_sd;

    Eigen::VectorXd P_mean;  // J
    Eigen::VectorXd P_sd;

    double alpha_mean;
    double alpha_sd;

    double alpha_zero_mean;
    double alpha_zero_sd;

    Eigen::MatrixXd mu_star_1_J_mean;  // J x G
    Eigen::MatrixXd mu_star_1_J_sd;

    Eigen::MatrixXd phi_star_1_J_mean;  // J x G
    Eigen::MatrixXd phi_star_1_J_sd;

    std::vector<Eigen::VectorXd> Beta_mean;  // D datasets
    std::vector<Eigen::VectorXd> Beta_sd;

    // Consensus cluster assignments (most frequent across chains)
    std::vector<std::vector<int>> Z_consensus;

    // Co-clustering matrix for each dataset (probability two cells in same cluster)
    std::vector<Eigen::MatrixXd> coclustering_matrices;

    // Overall acceptance rates (averaged across chains)
    double P_accept_mean;
    double alpha_accept_mean;
    double alpha_zero_accept_mean;
    double unique_accept_mean;
    double Beta_accept_mean;

    // Dimensions
    int D;
    std::vector<int> C;
    int G;
    int J;

    // Number of chains
    int num_chains;
};

// Run ensemble MCMC: multiple chains in parallel, then average
EnsembleResult ensemble_mcmc(
    const std::vector<Eigen::MatrixXd>& Y,
    int J,
    int num_chains = 100,
    int chain_length = 100,
    int thinning = 1,
    bool empirical = true,
    int burn_in = 30,
    bool quadratic = false,
    int iter_update = 10,
    double beta_mean = 0.06,
    double alpha_mu_2 = -1.0,
    double adaptive_prop = 0.1,
    bool print_progress = true,
    int num_cores = -1,  // -1 means use all available
    const Eigen::VectorXd& baynorm_mu_estimate = Eigen::VectorXd(),
    const Eigen::VectorXd& baynorm_phi_estimate = Eigen::VectorXd(),
    const std::vector<Eigen::VectorXd>& baynorm_beta = std::vector<Eigen::VectorXd>()
);

#endif // ENSEMBLE_MCMC_H