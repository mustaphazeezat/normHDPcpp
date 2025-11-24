#ifndef UNIQUE_PARAMETERS_H
#define UNIQUE_PARAMETERS_H

#include <Eigen/Dense>
#include <vector>

// Structure for unique parameters MCMC result
struct UniqueParamsResult {
    Eigen::MatrixXd mu_star_1_J_new;
    Eigen::MatrixXd phi_star_1_J_new;
    int accept_count;
    std::vector<std::vector<Eigen::MatrixXd>> tilde_s_mu_phi_new;
    std::vector<std::vector<Eigen::RowVectorXd>> mean_X_mu_phi_new;
    std::vector<std::vector<Eigen::MatrixXd>> covariance_new;
};

// Log probability for unique parameters
double unique_parameters_log_prob(
    double mu_star,
    double phi_star,
    const std::vector<std::vector<int>>& Z,
    const Eigen::VectorXd& b,
    double alpha_phi_2,
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<Eigen::VectorXd>& Beta,
    int j,
    int g,
    double alpha_mu_2,
    bool quadratic = false
);

// MCMC update for unique parameters (mu_star, phi_star)
UniqueParamsResult unique_parameters_mcmc(
    const Eigen::MatrixXd& mu_star_1_J,
    const Eigen::MatrixXd& phi_star_1_J,
    const std::vector<std::vector<Eigen::RowVectorXd>>& mean_X_mu_phi,
    const std::vector<std::vector<Eigen::MatrixXd>>& tilde_s_mu_phi,
    const std::vector<std::vector<int>>& Z,
    const Eigen::VectorXd& b,
    double alpha_phi_2,
    const std::vector<Eigen::VectorXd>& Beta,
    double alpha_mu_2,
    const std::vector<std::vector<Eigen::MatrixXd>>& covariance,
    int iter_num,
    bool quadratic,
    const std::vector<Eigen::MatrixXd>& Y,
    double adaptive_prop = 0.01,
    int num_cores = 1
);

#endif // UNIQUE_PARAMETERS_H