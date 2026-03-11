#ifndef INFERENCE_H
#define INFERENCE_H

#include <RcppArmadillo.h>
#include <RcppThread.h>

#include "gibbs.h"
#include "evil.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppThread)]]

using namespace Rcpp;

List EvolutionaryVI(
    arma::field<arma::mat> x, int k, arma::vec n, int N, int p,
    arma::vec discrete_fields, int n_discrete_fields, arma::vec M,
    arma::vec continuous_fields, int n_continuous_fields, arma::mat hyper_beta,
    arma::mat hyper_sigma, arma::vec hyper_phi, arma::vec hyper_tau,
    arma::vec hyper_delta, double overlap_prob, int n_parents = 10,
    int n_children = 50, double tol_cavi = 1e-5, int max_iter_cavi = 10,
    double tol_evi = 1e-5, int max_iter_evi = 50, bool verbose = true,
    int n_threads = 1, double max_time = 86400, bool custom_initializer = false,
    bool use_checkpoint = false, Nullable<List> initial_values = R_NilValue,
    Nullable<List> checkpoint_values = R_NilValue);

List MCMC(arma::field<arma::mat> x, int k, arma::vec n, int N, int p,
          arma::vec discrete_fields, int n_discrete_fields, arma::vec M,
          arma::vec continuous_fields, int n_continuous_fields,
          arma::mat hyper_beta, arma::mat hyper_sigma, arma::vec hyper_phi,
          arma::vec hyper_tau, arma::vec hyper_delta, int n_burnin, int n_gibbs,
          int n_split_merge, bool verbose = true, double max_time = 86400,
          bool custom_initializer = false, bool use_checkpoint = false,
          Nullable<List> initial_values = R_NilValue,
          Nullable<List> checkpoint_values = R_NilValue);

List CoordinateAscentVI(
    arma::field<arma::mat> x, int k, arma::vec n, int N, int p,
    arma::vec discrete_fields, int n_discrete_fields, arma::vec M,
    arma::vec continuous_fields, int n_continuous_fields, arma::mat hyper_beta,
    arma::mat hyper_sigma, arma::vec hyper_phi, arma::vec hyper_tau,
    arma::vec hyper_delta, double overlap_prob, double tol_cavi = 1e-5,
    int max_iter_cavi = 10, bool verbose = true, double max_time = 86400,
    bool custom_initializer = false, bool use_checkpoint = false,
    Nullable<List> initial_values = R_NilValue,
    Nullable<List> checkpoint_values = R_NilValue);

List DIG(arma::field<arma::mat> x, int k, arma::vec n, int N, int p,
         arma::vec discrete_fields, int n_discrete_fields, arma::vec M,
         arma::vec continuous_fields, int n_continuous_fields,
         arma::mat hyper_beta, arma::mat hyper_sigma, arma::vec hyper_phi,
         arma::vec hyper_tau, arma::vec hyper_delta,
         double decaying_upper_bound, int n_burnin, int n_gibbs, int batch_size,
         int n_epochs, double max_time, bool batch_update = true,
         bool verbose = true);

arma::mat psm_vi(arma::field<arma::mat> probs_field);

arma::mat psm_mcmc(arma::mat samples);

arma::imat flatten_posterior_samples(
    arma::field<arma::field<IntegerVector>> samples, int k, int N);

List evaluate_performance(arma::field<IntegerVector> estimation,
                          arma::field<IntegerVector> truth, int N,
                          bool return_matrix = false);

#endif
