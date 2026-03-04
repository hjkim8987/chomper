#ifndef CAVI_H
#define CAVI_H

#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

#include "utils.h"

using namespace Rcpp;

/******************************************************************************
    Functions for Evolutionary Variational Inference
        1. Initializers
        2. Utility Functions
        3. Updaters
        4. ELBO Calculator
******************************************************************************/

/*******************************************
    1. Initializers
      2. Utility Functions
      3. Updaters
      4. ELBO Calculator
*******************************************/

arma::colvec init_rho_ij(double a_l, double b_l, int n_i);

arma::field<arma::mat> init_gamma(const arma::field<arma::mat> &x,
                                  const arma::field<arma::vec> &linkage_index,
                                  int k, arma::vec n, int N, arma::vec M,
                                  arma::vec discrete_fields,
                                  int n_discrete_fields, arma::vec hyper_delta);

arma::mat init_y_mean(const arma::field<arma::mat> &x,
                      const arma::field<arma::vec> &linkage_index, int k,
                      arma::vec n, int N, double N_double,
                      arma::vec continuous_fields, int n_continuous_fields);

arma::mat init_y_var(const arma::field<arma::mat> &x, int k, arma::vec n, int N,
                     double N_double, arma::vec continuous_fields,
                     int n_continuous_fields);

arma::vec init_sigma_scale(arma::mat hyper_sigma, arma::vec sigma_shape,
                           arma::mat eta_tilde, arma::mat sigma_tilde,
                           double N_double, int n_continuous_fields);

arma::field<arma::mat> init_nu(arma::field<arma::vec> index, int k, arma::vec n,
                               int N);

/*******************************************
      1. Initializers
    2. Utility Functions
      3. Updaters
      4. ELBO Calculator
*******************************************/

arma::vec sum_ij_rho_ijl(const arma::field<arma::mat> &rho, int k,
                         double N_double, int p, bool complement = false);

double log_beta_constant(arma::vec x, int n);

double beta_function(double a, double b, bool log = false);

/*******************************************
      1. Initializers
      2. Utility Functions
    3. Updaters
      4. ELBO Calculator
*******************************************/

arma::field<arma::vec> update_alpha(const arma::field<arma::mat> &x,
                                    const arma::field<arma::vec> &mu,
                                    const arma::field<arma::mat> &gamma,
                                    const arma::field<arma::mat> &rho, int k,
                                    arma::vec discrete_fields,
                                    int n_discrete_fields, arma::vec M);

arma::field<arma::vec> update_psi(const arma::field<arma::vec> &alpha,
                                  int n_discrete_fields, arma::vec M);

arma::vec update_eta_mean(const arma::field<arma::mat> &x,
                          const arma::field<arma::mat> &rho,
                          arma::mat eta_tilde, arma::vec continuous_fields,
                          int n_continuous_fields, arma::mat rho_sum, int k,
                          double N_double);

arma::vec update_eta_var(arma::vec sigma_shape, arma::vec sigma_scale,
                         arma::vec continuous_fields, int n_continuous_fields,
                         arma::mat rho_sum, double N_double);

arma::vec update_sigma_shape(arma::mat hyper_sigma, arma::mat rho_sum,
                             arma::vec continuous_fields,
                             int n_continuous_fields, double N_double);

arma::vec update_sigma_scale(arma::mat hyper_sigma, arma::mat eta_tilde,
                             arma::mat sigma_tilde,
                             const arma::field<arma::mat> &x,
                             const arma::field<arma::mat> &rho, arma::mat zeta,
                             arma::vec continuous_fields,
                             int n_continuous_fields, int k);

arma::mat update_omega(arma::mat hyper_beta, arma::mat rho_sum, int p);

arma::mat update_kappa(arma::mat omega, int p);

arma::field<arma::field<arma::mat>> update_nu_all(
    const arma::field<arma::mat> &x, const arma::field<arma::mat> &rho,
    const arma::field<arma::mat> &gamma, arma::mat eta_tilde,
    arma::mat sigma_tilde, int k, arma::vec n, int N, arma::vec hyper_tau,
    arma::vec log_phi_tau, arma::vec discrete_fields, int n_discrete_fields,
    arma::vec M, arma::vec hyper_delta, arma::vec continuous_fields,
    int n_continuous_fields);

arma::field<arma::field<arma::mat>> update_gamma(
    const arma::field<arma::mat> &x, const arma::field<arma::mat> &rho,
    const arma::field<arma::mat> &nu, const arma::field<arma::vec> &psi, int k,
    arma::vec n, int N, arma::vec discrete_fields, int n_discrete_fields,
    arma::vec M, arma::vec hyper_tau, arma::vec hyper_delta,
    arma::vec log_phi_tau);

arma::field<arma::mat> update_chi_discrete(
    const arma::field<arma::mat> &x, const arma::field<arma::mat> &gamma,
    const arma::field<arma::mat> &nu, int k, arma::vec n, arma::vec M, int N,
    arma::vec discrete_fields, int n_discrete_fields, arma::vec hyper_delta);

arma::field<arma::mat> update_continuous_y(
    const arma::field<arma::mat> &x, const arma::field<arma::mat> &rho,
    const arma::field<arma::mat> &nu, arma::vec eta_mean, arma::vec sigma_shape,
    arma::vec sigma_scale, arma::vec hyper_tau, int k, arma::vec n, int N,
    arma::vec continuous_fields, int n_continuous_fields);

arma::field<arma::mat> update_chi_expectation(const arma::field<arma::mat> &nu,
                                              arma::mat eta_tilde, int k,
                                              arma::vec n,
                                              int n_continuous_fields);

arma::field<arma::mat> update_chi_squared_expectation(
    const arma::field<arma::mat> &nu, arma::mat eta_tilde,
    arma::mat sigma_tilde, int k, arma::vec n, int n_continuous_fields);

arma::mat update_zeta(const arma::field<arma::mat> &x,
                      const arma::field<arma::mat> &rho, arma::mat eta_tilde,
                      arma::vec eta_var, arma::vec sigma_shape,
                      arma::vec sigma_scale, arma::mat rho_sum,
                      arma::vec continuous_fields, int n_continuous_fields,
                      int k, double N_double, bool before_sigma = true);

arma::field<arma::mat> update_rho(
    const arma::field<arma::mat> &x, int k, arma::vec n, int p,
    arma::vec discrete_fields, int n_discrete_fields, arma::vec M,
    const arma::field<arma::mat> &log_constant_matrix, arma::vec sigma_shape,
    arma::vec sigma_scale, const arma::field<arma::vec> &psi,
    const arma::field<arma::mat> &chi_discrete, arma::vec log_phi_tau,
    arma::vec continuous_fields, int n_continuous_fields,
    const arma::field<arma::mat> &chi_expectation,
    const arma::field<arma::mat> &chi_squared_expectation, arma::mat zeta,
    arma::vec hyper_tau, arma::mat kappa);

/*******************************************
      1. Initializers
      2. Utility Functions
      3. Updaters
    4. ELBO Calculator
*******************************************/

double calculate_ELBO(
    const arma::field<arma::mat> &x, const arma::field<arma::mat> &rho,
    arma::mat rho_sum, const arma::field<arma::mat> &nu,
    const arma::field<arma::mat> &log_nu, const arma::field<arma::mat> &gamma,
    const arma::field<arma::mat> &log_gamma,
    const arma::field<arma::vec> &alpha, const arma::field<arma::vec> &psi,
    arma::mat omega, arma::mat eta_tilde, arma::mat sigma_tilde,
    arma::vec sigma_shape, arma::vec sigma_scale,
    const arma::field<arma::mat> &chi_discrete,
    const arma::field<arma::mat> &chi_expectation,
    const arma::field<arma::mat> &chi_squared_expectation, arma::mat zeta,
    const arma::field<arma::vec> &hyper_mu, arma::vec hyper_tau,
    arma::mat hyper_sigma, arma::vec log_phi_tau,
    const arma::field<arma::mat> &log_constant_matrix, int k, arma::vec n,
    int N, double N_double, arma::vec discrete_fields, int n_discrete_fields,
    arma::vec M, arma::vec continuous_fields, int n_continuous_fields);

#endif