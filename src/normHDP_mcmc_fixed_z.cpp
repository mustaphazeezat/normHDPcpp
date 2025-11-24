// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

// Adjust these paths based on your actual include structure
#include "includes/dataset_specific.h"
#include "includes/component_probabilities.h"
#include "includes/alpha.h"
#include "includes/alpha_zero.h"
#include "includes/unique_parameters.h"
#include "includes/capture_efficiencies.h"
#include "includes/mean_dispersion.h"
#include "includes/utils.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <set>
#include <map>

// [[Rcpp::export]]
Rcpp::List normHDP_mcmc_fixed_z(
    const std::vector<Eigen::MatrixXd>& Y,
    const std::vector<std::vector<int>>& cluster_estimates,
    int number_iter,
    int thinning = 5,
    bool empirical = true,
    int burn_in = 3000,
    bool quadratic = false,
    int iter_update = 100,
    double beta_mean = 0.06,
    double alpha_mu_2 = -1.0,
    double adaptive_prop = 0.1,
    int num_cores = 1,
    SEXP baynorm_mu_estimate = R_NilValue,
    SEXP baynorm_phi_estimate = R_NilValue,
    SEXP baynorm_beta = R_NilValue
) {

    // ============ Input Validation ============
    if (Y.empty()) {
        Rcpp::stop("Y is empty");
    }
    if (cluster_estimates.empty()) {
        Rcpp::stop("cluster_estimates is empty");
    }
    if (Y.size() != cluster_estimates.size()) {
        Rcpp::stop("Y and cluster_estimates must have same length (number of datasets)");
    }

    // ============ Dimensions ============
    int D = Y.size();
    int G = Y[0].rows();
    std::vector<int> C(D);

    Rcpp::Rcout << "Checking dimensions..." << std::endl;

    for (int d = 0; d < D; ++d) {
        C[d] = Y[d].cols();

        if (Y[d].rows() != G) {
            Rcpp::stop("All datasets must have same number of genes. Dataset 0 has %d genes, dataset %d has %d genes",
                       G, d, Y[d].rows());
        }
        if ((int)cluster_estimates[d].size() != C[d]) {
            Rcpp::stop("Dataset %d: cluster_estimates size (%d) != number of cells (%d)",
                       d, cluster_estimates[d].size(), C[d]);
        }
    }

    // ============ Map clusters to 0-based sequential indices ============
    std::vector<std::vector<int>> Z;

    // Collect all unique cluster labels
    std::set<int> unique_clusters;
    for (const auto& z_d : cluster_estimates) {
        unique_clusters.insert(z_d.begin(), z_d.end());
    }

    // Create mapping from original labels to 0-based indices
    std::map<int, int> cluster_map;
    int idx = 0;
    for (int orig_label : unique_clusters) {
        cluster_map[orig_label] = idx++;
    }

    // Apply mapping
    Z.resize(D);
    for (int d = 0; d < D; ++d) {
        Z[d].resize(cluster_estimates[d].size());
        for (size_t c = 0; c < cluster_estimates[d].size(); ++c) {
            Z[d][c] = cluster_map[cluster_estimates[d][c]];
        }
    }

    // Determine J from the number of unique clusters
    int J = unique_clusters.size();

    Rcpp::Rcout << "\nnormHDP with Fixed Z" << std::endl;
    Rcpp::Rcout << std::string(60, '=') << std::endl;
    Rcpp::Rcout << "Datasets (D): " << D << std::endl;
    Rcpp::Rcout << "Genes (G): " << G << std::endl;
    Rcpp::Rcout << "Clusters (J): " << J << std::endl;
    Rcpp::Rcout << "Total cells: " << std::accumulate(C.begin(), C.end(), 0) << std::endl;
    Rcpp::Rcout << "Iterations: " << number_iter << ", Burn-in: " << burn_in
                << ", Thinning: " << thinning << std::endl;
    Rcpp::Rcout << "Z is FIXED (not updated during MCMC)" << std::endl;
    Rcpp::Rcout << std::string(60, '=') << std::endl << std::endl;

    // ============ BayNorm Estimates - Convert from SEXP ============
    Rcpp::NumericVector baynorm_mu_vec(baynorm_mu_estimate);
    Rcpp::NumericVector baynorm_phi_vec(baynorm_phi_estimate);
    Rcpp::List baynorm_beta_list(baynorm_beta);

    if (baynorm_mu_vec.size() != G) {
        Rcpp::stop("baynorm_mu_estimate length (%d) != number of genes (%d)",
                   baynorm_mu_vec.size(), G);
    }
    if (baynorm_phi_vec.size() != G) {
        Rcpp::stop("baynorm_phi_estimate length (%d) != number of genes (%d)",
                   baynorm_phi_vec.size(), G);
    }
    if (baynorm_beta_list.size() != D) {
        Rcpp::stop("baynorm_beta list length (%d) != number of datasets (%d)",
                   baynorm_beta_list.size(), D);
    }

    Eigen::VectorXd mu_estimate = Rcpp::as<Eigen::VectorXd>(baynorm_mu_vec);
    Eigen::VectorXd phi_estimate = Rcpp::as<Eigen::VectorXd>(baynorm_phi_vec);

    std::vector<Eigen::VectorXd> Beta_estimate(D);
    for (int d = 0; d < D; ++d) {
        Beta_estimate[d] = Rcpp::as<Eigen::VectorXd>(baynorm_beta_list[d]);
        if (Beta_estimate[d].size() != C[d]) {
            Rcpp::stop("baynorm_beta[%d] length (%d) != number of cells (%d) in dataset %d",
                       d, Beta_estimate[d].size(), C[d], d);
        }
    }

    // Clean estimates - replace non-finite with small positive values
    for (int g = 0; g < G; ++g) {
        if (mu_estimate(g) <= 0 || !std::isfinite(mu_estimate(g))) {
            mu_estimate(g) = 0.01;
        }
        if (phi_estimate(g) <= 0 || !std::isfinite(phi_estimate(g))) {
            phi_estimate(g) = 0.01;
        }
    }

    for (int d = 0; d < D; ++d) {
        for (int c = 0; c < C[d]; ++c) {
            if (Beta_estimate[d](c) <= 0 || Beta_estimate[d](c) >= 1 ||
                !std::isfinite(Beta_estimate[d](c))) {
                Beta_estimate[d](c) = 0.06;
            }
        }
    }

    Rcpp::Rcout << "BayNorm estimates validated and cleaned" << std::endl;

    // ============ Hyper-parameters ============

    // alpha_mu_2
    if (alpha_mu_2 < 0) {
        double sum_log_mu = 0.0;
        int count = 0;
        for (int g = 0; g < G; ++g) {
            if (std::isfinite(std::log(mu_estimate(g)))) {
                sum_log_mu += mu_estimate(g);
                count++;
            }
        }
        if (count > 0) {
            alpha_mu_2 = 2.0 * std::log(sum_log_mu / count);
        } else {
            alpha_mu_2 = 1.0;  // default
        }
    }

    // a_d_beta, b_d_beta (Beta distribution parameters for capture efficiency prior)
    Eigen::VectorXd a_d_beta(D);
    Eigen::VectorXd b_d_beta(D);

    for (int d = 0; d < D; ++d) {
        double baynorm_mean_capeff = beta_mean;
        double baynorm_var_capeff = 0.5;
        double a_beta, b_beta;

        int max_iterations = 100;
        int iter = 0;

        do {
            a_beta = ((1.0 - baynorm_mean_capeff) / baynorm_var_capeff -
                     1.0 / baynorm_mean_capeff) * baynorm_mean_capeff * baynorm_mean_capeff;
            b_beta = a_beta * (1.0 / baynorm_mean_capeff - 1.0);

            if (baynorm_var_capeff >= baynorm_mean_capeff * (1.0 - baynorm_mean_capeff) ||
                a_beta < 1.0 || b_beta < 1.0) {
                baynorm_var_capeff /= 2.0;
                iter++;
                if (iter > max_iterations) {
                    Rcpp::stop("Could not find valid a_beta, b_beta parameters after %d iterations", max_iterations);
                }
            } else {
                break;
            }
        } while (true);

        a_d_beta(d) = a_beta;
        b_d_beta(d) = b_beta;
    }

    // Regression parameters for mu-phi relationship
    double v_1, v_2;
    Eigen::VectorXd m_b;

    std::vector<double> x_vals, y_vals;
    for (int g = 0; g < G; ++g) {
        if (std::isfinite(std::log(mu_estimate(g))) &&
            std::isfinite(std::log(phi_estimate(g)))) {
            x_vals.push_back(std::log(mu_estimate(g)));
            y_vals.push_back(std::log(phi_estimate(g)));
        }
    }

    int n_valid = x_vals.size();
    if (n_valid < 10) {
        Rcpp::stop("Too few valid (mu, phi) pairs for regression: %d", n_valid);
    }

    Eigen::VectorXd y_vec(n_valid);
    Eigen::MatrixXd X_mat;

    for (int i = 0; i < n_valid; ++i) {
        y_vec(i) = y_vals[i];
    }

    if (quadratic) {
        X_mat.resize(n_valid, 3);
        for (int i = 0; i < n_valid; ++i) {
            X_mat(i, 0) = 1.0;
            X_mat(i, 1) = x_vals[i];
            X_mat(i, 2) = x_vals[i] * x_vals[i];
        }
        m_b.resize(3);
        m_b << -1.0, 2.0, 0.0;
    } else {
        X_mat.resize(n_valid, 2);
        for (int i = 0; i < n_valid; ++i) {
            X_mat(i, 0) = 1.0;
            X_mat(i, 1) = x_vals[i];
        }
        m_b.resize(2);
        m_b << -1.0, 2.0;
    }

    Eigen::VectorXd coef = (X_mat.transpose() * X_mat).ldlt().solve(X_mat.transpose() * y_vec);
    Eigen::VectorXd resid = y_vec - X_mat * coef;
    double rse_squared = resid.squaredNorm() / (n_valid - coef.size());

    if (empirical) {
        m_b = coef;
        double variance = 5.0;
        v_1 = rse_squared * rse_squared / variance + 2.0;
        v_2 = (v_1 - 1.0) * rse_squared;
    } else {
        v_1 = 2.0;
        v_2 = 1.0;
    }

    Rcpp::Rcout << "Hyper-parameters initialized" << std::endl;

    // ============ Initial Values ============
    Eigen::VectorXd b_initial = m_b;
    double alpha_phi_2_initial = rse_squared;

    Eigen::MatrixXd P_J_D_initial = Eigen::MatrixXd::Constant(J, D, 1.0 / J);
    Eigen::VectorXd P_initial = Eigen::VectorXd::Constant(J, 1.0 / J);
    double alpha_initial = 1.0;
    double alpha_zero_initial = 1.0;

    Eigen::MatrixXd mu_star_1_J_initial(J, G);
    Eigen::MatrixXd phi_star_1_J_initial(J, G);
    for (int j = 0; j < J; ++j) {
        mu_star_1_J_initial.row(j) = mu_estimate.transpose();
        phi_star_1_J_initial.row(j) = phi_estimate.transpose();
    }

    // ============ Prepare Outputs ============
    if (burn_in >= number_iter) {
        Rcpp::stop("burn_in (%d) must be < number_iter (%d)", burn_in, number_iter);
    }

    int num_saved = 0;
    if (number_iter > burn_in) {
        num_saved = ((number_iter - burn_in - 1) / thinning) + 1;
    }

    if (num_saved <= 0) {
        Rcpp::stop("No samples to save with current parameters");
    }

    Rcpp::Rcout << "Will save " << num_saved << " posterior samples\n" << std::endl;

    std::vector<Eigen::VectorXd> b_output;
    std::vector<double> alpha_phi2_output;
    std::vector<Eigen::MatrixXd> P_J_D_output;
    std::vector<Eigen::VectorXd> P_output;
    std::vector<double> alpha_output_vec;
    std::vector<double> alpha_zero_output_vec;
    std::vector<Eigen::MatrixXd> mu_star_1_J_output;
    std::vector<Eigen::MatrixXd> phi_star_1_J_output;
    std::vector<std::vector<Eigen::VectorXd>> Beta_output;

    b_output.reserve(num_saved);
    alpha_phi2_output.reserve(num_saved);
    P_J_D_output.reserve(num_saved);
    P_output.reserve(num_saved);
    alpha_output_vec.reserve(num_saved);
    alpha_zero_output_vec.reserve(num_saved);
    mu_star_1_J_output.reserve(num_saved);
    phi_star_1_J_output.reserve(num_saved);
    Beta_output.reserve(num_saved);

    // Acceptance tracking
    std::vector<double> P_accept_vec(number_iter - 1);
    std::vector<double> alpha_accept_vec(number_iter - 1);
    std::vector<double> alpha_zero_accept_vec(number_iter - 1);
    std::vector<double> unique_accept_vec(number_iter - 1);
    std::vector<double> Beta_accept_vec(number_iter - 1);

    // ============ Set Initial Values as Current ============
    Eigen::VectorXd b_new = b_initial;
    double alpha_phi_2_new = alpha_phi_2_initial;
    Eigen::MatrixXd P_J_D_new = P_J_D_initial;
    Eigen::VectorXd P_new = P_initial;
    double alpha_new = alpha_initial;
    double alpha_zero_new = alpha_zero_initial;
    Eigen::MatrixXd mu_star_1_J_new = mu_star_1_J_initial;
    Eigen::MatrixXd phi_star_1_J_new = phi_star_1_J_initial;
    std::vector<Eigen::VectorXd> Beta_new = Beta_estimate;

    int P_count = 0, alpha_count = 0, alpha_zero_count = 0;
    int unique_count = 0, Beta_count = 0;

    // ============ Covariance Structures ============

    Eigen::RowVectorXd mean_X_component_new(J - 1);
    for (int j = 0; j < J - 1; ++j) {
        mean_X_component_new(j) = std::log(P_initial(j) / P_initial(J - 1));
    }
    Eigen::MatrixXd tilde_s_component_new = mean_X_component_new.transpose() * mean_X_component_new;
    Eigen::MatrixXd covariance_component_new = Eigen::MatrixXd::Zero(J - 1, J - 1);

    double mean_X_alpha_new = std::log(alpha_new);
    double M_2_alpha_new = 0.0;
    double variance_alpha_new = 0.0;

    double mean_X_alpha_zero_new = std::log(alpha_zero_new);
    double M_2_alpha_zero_new = 0.0;
    double variance_alpha_zero_new = 0.0;

    std::vector<std::vector<Eigen::MatrixXd>> covariance_unique_new(J);
    std::vector<std::vector<Eigen::MatrixXd>> tilde_s_unique_new(J);
    std::vector<std::vector<Eigen::RowVectorXd>> mean_X_unique_new(J);

    for (int j = 0; j < J; ++j) {
        covariance_unique_new[j].resize(G);
        tilde_s_unique_new[j].resize(G);
        mean_X_unique_new[j].resize(G);

        for (int g = 0; g < G; ++g) {
            covariance_unique_new[j][g] = Eigen::Matrix2d::Zero();

            Eigen::Vector2d x_vec;
            x_vec << std::log(mu_star_1_J_new(j, g)), std::log(phi_star_1_J_new(j, g));
            tilde_s_unique_new[j][g] = x_vec * x_vec.transpose();
            mean_X_unique_new[j][g] = x_vec.transpose();
        }
    }

    std::vector<Eigen::VectorXd> variance_capture_new(D);
    std::vector<Eigen::VectorXd> mean_X_capture_new(D);
    std::vector<Eigen::VectorXd> M_2_capture_new(D);

    for (int d = 0; d < D; ++d) {
        variance_capture_new[d] = Eigen::VectorXd::Zero(C[d]);
        M_2_capture_new[d] = Eigen::VectorXd::Zero(C[d]);
        mean_X_capture_new[d].resize(C[d]);
        for (int c = 0; c < C[d]; ++c) {
            mean_X_capture_new[d](c) = std::log(Beta_new[d](c) / (1.0 - Beta_new[d](c)));
        }
    }

    Rcpp::Rcout << "Starting MCMC iterations..." << std::endl << std::endl;

    // ============ MCMC Iterations ============
    for (int iter = 2; iter <= number_iter; ++iter) {

        if (iter % iter_update == 0) {
            Rcpp::Rcout << "Iteration: " << iter << " / " << number_iter << std::endl;
        }

        // Check for user interrupt every 100 iterations
        if (iter % 100 == 0) {
            Rcpp::checkUserInterrupt();
        }

        try {
            // 1) Regression parameters
            auto mean_disp_output = mean_dispersion_mcmc(mu_star_1_J_new, phi_star_1_J_new,
                                                         v_1, v_2, m_b, quadratic);
            alpha_phi_2_new = mean_disp_output.alpha_phi_2;
            b_new = mean_disp_output.b;
        } catch (std::exception& e) {
            Rcpp::stop("Error in mean_dispersion_mcmc at iteration %d: %s", iter, e.what());
        }

        try {
            // 2) Dataset-specific probabilities (Z is fixed)
            P_J_D_new = dataset_specific_mcmc(Z, P_new, alpha_new);
        } catch (std::exception& e) {
            Rcpp::stop("Error in dataset_specific_mcmc at iteration %d: %s", iter, e.what());
        }

        try {
            // 3) Component probabilities
            auto comp_output = component_probabilities_mcmc(P_new, P_J_D_new, alpha_zero_new,
                                                           alpha_new, covariance_component_new,
                                                           mean_X_component_new, tilde_s_component_new,
                                                           iter, adaptive_prop);
            P_new = comp_output.P_new;
            tilde_s_component_new = comp_output.tilde_s_new;
            mean_X_component_new = comp_output.mean_x_new;
            covariance_component_new = comp_output.covariance_new;
            P_count += comp_output.accept;
            P_accept_vec[iter - 2] = (double)P_count / (iter - 1);
        } catch (std::exception& e) {
            Rcpp::stop("Error in component_probabilities_mcmc at iteration %d: %s", iter, e.what());
        }

        try {
            // 4) Alpha
            auto alpha_output = alpha_mcmc(P_J_D_new, P_new, alpha_new, mean_X_alpha_new,
                                          M_2_alpha_new, variance_alpha_new, iter, adaptive_prop);
            alpha_new = alpha_output.alpha_new;
            mean_X_alpha_new = alpha_output.X_mean_new;
            M_2_alpha_new = alpha_output.M_2_new;
            variance_alpha_new = alpha_output.variance_new;
            alpha_count += alpha_output.accept;
            alpha_accept_vec[iter - 2] = (double)alpha_count / (iter - 1);
        } catch (std::exception& e) {
            Rcpp::stop("Error in alpha_mcmc at iteration %d: %s", iter, e.what());
        }

        try {
            // 5) Alpha zero
            auto alpha_zero_output = alpha_zero_mcmc(P_new, alpha_zero_new, mean_X_alpha_zero_new,
                                                    M_2_alpha_zero_new, variance_alpha_zero_new,
                                                    iter, adaptive_prop);
            alpha_zero_new = alpha_zero_output.alpha_zero_new;
            mean_X_alpha_zero_new = alpha_zero_output.X_mean_new;
            M_2_alpha_zero_new = alpha_zero_output.M_2_new;
            variance_alpha_zero_new = alpha_zero_output.variance_new;
            alpha_zero_count += alpha_zero_output.accept;
            alpha_zero_accept_vec[iter - 2] = (double)alpha_zero_count / (iter - 1);
        } catch (std::exception& e) {
            Rcpp::stop("Error in alpha_zero_mcmc at iteration %d: %s", iter, e.what());
        }

        try {
            // 6) Unique parameters
            auto unique_output = unique_parameters_mcmc(mu_star_1_J_new, phi_star_1_J_new,
                                                       mean_X_unique_new, tilde_s_unique_new,
                                                       Z, b_new, alpha_phi_2_new, Beta_new,
                                                       alpha_mu_2, covariance_unique_new, iter,
                                                       quadratic, Y, adaptive_prop, num_cores);
            mu_star_1_J_new = unique_output.mu_star_1_J_new;
            phi_star_1_J_new = unique_output.phi_star_1_J_new;
            tilde_s_unique_new = unique_output.tilde_s_mu_phi_new;
            mean_X_unique_new = unique_output.mean_X_mu_phi_new;
            covariance_unique_new = unique_output.covariance_new;
            unique_count += unique_output.accept_count;
            unique_accept_vec[iter - 2] = (double)unique_count / ((iter - 1) * J * G);
        } catch (std::exception& e) {
            Rcpp::stop("Error in unique_parameters_mcmc at iteration %d: %s", iter, e.what());
        }

        try {
            // 7) Capture efficiencies
            auto capture_output = capture_efficiencies_mcmc(Beta_new, Y, Z, mu_star_1_J_new,
                                                           phi_star_1_J_new, a_d_beta, b_d_beta,
                                                           iter, M_2_capture_new, mean_X_capture_new,
                                                           variance_capture_new, adaptive_prop);
            Beta_new = capture_output.Beta_new;
            mean_X_capture_new = capture_output.mean_X_new;
            M_2_capture_new = capture_output.M_2_new;
            variance_capture_new = capture_output.variance_new;
            Beta_count += capture_output.accept_count;
            int total_cells = std::accumulate(C.begin(), C.end(), 0);
            Beta_accept_vec[iter - 2] = (double)Beta_count / ((iter - 1) * total_cells);
        } catch (std::exception& e) {
            Rcpp::stop("Error in capture_efficiencies_mcmc at iteration %d: %s", iter, e.what());
        }

        // 8) Save outputs
        if (iter >= burn_in && (iter - burn_in) % thinning == 0) {
            try {
                b_output.push_back(b_new);
                alpha_phi2_output.push_back(alpha_phi_2_new);
                P_J_D_output.push_back(P_J_D_new);
                P_output.push_back(P_new);
                alpha_output_vec.push_back(alpha_new);
                alpha_zero_output_vec.push_back(alpha_zero_new);
                mu_star_1_J_output.push_back(mu_star_1_J_new);
                phi_star_1_J_output.push_back(phi_star_1_J_new);
                Beta_output.push_back(Beta_new);
            } catch (std::exception& e) {
                Rcpp::stop("Error saving output at iteration %d: %s", iter, e.what());
            }
        }
    }

    Rcpp::Rcout << "\nMCMC completed! Saved " << b_output.size() << " samples.\n" << std::endl;

    // ============ Create acceptance probability data frame ============
    Rcpp::DataFrame acceptance_prob_list = Rcpp::DataFrame::create(
        Rcpp::Named("P_accept") = P_accept_vec,
        Rcpp::Named("alpha_accept") = alpha_accept_vec,
        Rcpp::Named("alpha_zero_accept") = alpha_zero_accept_vec,
        Rcpp::Named("unique_accept") = unique_accept_vec,
        Rcpp::Named("Beta_accept") = Beta_accept_vec
    );

    // ============ Return as Rcpp::List ============
    return Rcpp::List::create(
        Rcpp::Named("b_output") = b_output,
        Rcpp::Named("alpha_phi2_output") = alpha_phi2_output,
        Rcpp::Named("P_J_D_output") = P_J_D_output,
        Rcpp::Named("P_output") = P_output,
        Rcpp::Named("alpha_output") = alpha_output_vec,
        Rcpp::Named("alpha_zero_output") = alpha_zero_output_vec,
        Rcpp::Named("mu_star_1_J_output") = mu_star_1_J_output,
        Rcpp::Named("phi_star_1_J_output") = phi_star_1_J_output,
        Rcpp::Named("Beta_output") = Beta_output,
        Rcpp::Named("acceptance_prob_list") = acceptance_prob_list,
        Rcpp::Named("J") = J,
        Rcpp::Named("D") = D,
        Rcpp::Named("C") = C,
        Rcpp::Named("G") = G,
        Rcpp::Named("Z") = cluster_estimates  // Return original cluster labels
    );
}