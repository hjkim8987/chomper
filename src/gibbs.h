#ifndef GIBBS_H
#define GIBBS_H

#include <RcppArmadillo.h>

#include "utils.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

double rbeta_l(double a, double b, arma::field<arma::mat> z, int l, int k,
               int N);
NumericVector rtheta_l(NumericVector mu, arma::field<arma::mat> x, arma::mat y,
                       arma::field<arma::mat> z, int l, int k, int M_l);

int ry_jprimel_multinomial(double phi_l, double tau_l, arma::field<arma::mat> x,
                           arma::field<IntegerVector> lambda,
                           arma::field<arma::mat> z, NumericVector theta_l,
                           int jprime, int l, int k, arma::vec n, int M_l,
                           arma::field<arma::field<arma::mat>> x_in_range,
                           int l_original);

double ry_jprimel_normal(double tau_l, int jprime, int l, int k, arma::vec n,
                         arma::field<arma::mat> x, arma::field<arma::mat> z,
                         arma::field<IntegerVector> lambda, double eta_l,
                         double sigma_l);

double reta_l(arma::field<arma::mat> x, arma::field<arma::mat> z,
              arma::colvec y_l, double sigma_l, int N, int l, int k);

double rsigma_l(arma::rowvec hyper_sigma_l, arma::colvec y_l, double eta_l,
                arma::field<arma::mat> x, arma::field<arma::mat> z, int l,
                int k, int N);

int rz_ijl_multinomial(double beta_l, double theta_l, double x_probability_yx);

int rz_ijl_normal(double tau_l, double x_ijl, double beta_l, double y_jprimel,
                  double eta_l, double sigma_l);

double get_loglikelihood(arma::rowvec z, int lambda_ij, int j1,
                         arma::rowvec y_j1, int j2, arma::rowvec y_j2,

                         arma::rowvec x,
                         arma::field<arma::mat> log_x_probability,

                         arma::field<NumericVector> log_theta,
                         NumericVector log_odds, arma::field<NumericVector> mu,

                         NumericVector eta, NumericVector sigma,
                         arma::vec hyper_tau, arma::vec discrete_fields,
                         int n_discrete_fields, arma::vec M,
                         arma::vec continuous_fields, int n_continuous_fields);

double get_loglikelihood_latent(
    arma::rowvec y, arma::field<NumericVector> log_theta, NumericVector eta,
    NumericVector sigma, arma::vec discrete_fields, int n_discrete_fields,
    arma::vec continuous_fields, int n_continuous_fields, arma::vec M);

arma::mat create_split_collection(IntegerVector file_index,
                                  IntegerVector location_index,
                                  IntegerVector file_selected,
                                  IntegerVector location_selected, int j1,
                                  int j2);

arma::rowvec update_distortion(arma::rowvec x_ij, arma::rowvec y_jprime,
                               NumericVector beta,
                               arma::field<NumericVector> theta,
                               NumericVector eta, NumericVector sigma, int p,
                               arma::vec discrete_fields, int n_discrete_fields,
                               arma::vec continuous_fields,
                               int n_continuous_fields, arma::vec hyper_tau,
                               arma::field<arma::mat> x_probability);

arma::rowvec update_latent_record(
    arma::vec hyper_phi, arma::vec hyper_tau, arma::field<arma::mat> x,
    arma::field<IntegerVector> lambda, arma::field<arma::mat> z,
    arma::field<NumericVector> theta, NumericVector eta, NumericVector sigma,
    int jprime, int k, arma::vec n, arma::vec M, int p,
    arma::vec discrete_fields, int n_discrete_fields,
    arma::vec continuous_fields, int n_continuous_fields,
    arma::field<arma::field<arma::mat>> x_in_range);

arma::field<arma::mat> do_split(
    int lambda_ij,

    IntegerVector selected_files, IntegerVector selected_records_index,
    IntegerVector entire_index, int k, arma::vec n, int p, arma::vec M,

    arma::field<arma::mat> x, arma::mat y, arma::field<arma::mat> z,
    arma::field<IntegerVector> lambda, NumericVector beta,
    arma::field<NumericVector> theta, NumericVector eta, NumericVector sigma,
    NumericVector log_odds, arma::field<NumericVector> log_theta,

    arma::vec discrete_fields, int n_discrete_fields,
    arma::vec continuous_fields, int n_continuous_fields, arma::vec hyper_phi,
    arma::vec hyper_tau, arma::field<NumericVector> mu,
    arma::field<arma::mat> x_probability,
    arma::field<arma::mat> log_x_probability,
    arma::field<arma::field<arma::mat>> x_in_range, bool sample_y = false);

arma::field<arma::mat> do_merge(
    int lambda1, int lambda2,

    IntegerVector selected_files, IntegerVector selected_records_index, int k,
    arma::vec n, int p, arma::vec M,

    arma::field<arma::mat> x, arma::mat y, arma::field<arma::mat> z,
    arma::field<IntegerVector> lambda, NumericVector beta,
    arma::field<NumericVector> theta, NumericVector eta, NumericVector sigma,
    NumericVector log_odds, arma::field<NumericVector> log_theta,

    arma::vec discrete_fields, int n_discrete_fields,
    arma::vec continuous_fields, int n_continuous_fields, arma::vec hyper_phi,
    arma::vec hyper_tau, arma::field<NumericVector> mu,
    arma::field<arma::mat> x_probability,
    arma::field<arma::mat> log_x_probability,
    arma::field<arma::field<arma::mat>> x_in_range, bool sample_y = false);

arma::field<arma::mat> do_split_merge(
    IntegerVector files, arma::field<IntegerVector> indices,
    IntegerVector entire_index, int k, arma::vec n, int p, arma::vec M,

    arma::field<arma::mat> x, arma::mat y, arma::field<arma::mat> z,
    arma::field<IntegerVector> lambda, NumericVector beta,
    arma::field<NumericVector> theta, NumericVector eta, NumericVector sigma,
    NumericVector log_odds, arma::field<NumericVector> log_theta,

    arma::vec discrete_fields, int n_discrete_fields,
    arma::vec continuous_fields, int n_continuous_fields, arma::vec hyper_phi,
    arma::vec hyper_tau, arma::field<NumericVector> mu,
    arma::field<arma::mat> x_probability,
    arma::field<arma::mat> log_x_probability,
    arma::field<arma::field<arma::mat>> x_in_range, bool sample_y = false);

/**************************************************/
// For Discomfort-Informed Adaptive Gibbs Sampler
/**************************************************/
double get_categorical_loglikelihood(
    const arma::rowvec& x_ij, const arma::rowvec& y_jprime,
    const arma::rowvec& z_ij, const arma::field<NumericVector>& log_theta,
    const arma::field<arma::mat>& log_x_probability,
    const arma::vec& discrete_fields, int n_discrete_fields);

double get_gaussian_loglikelihood(
    const arma::rowvec& x_ij, const arma::rowvec& y_jprime,
    const arma::rowvec& z_ij, const NumericVector& eta,
    const NumericVector& sigma, const arma::vec& hyper_tau,
    const arma::vec& continuous_fields, int n_continuous_fields);

arma::field<arma::mat> get_likelihood(
    const arma::field<arma::mat>& x, const arma::mat& y,
    const arma::field<arma::mat>& z, const arma::field<IntegerVector>& lambda,
    const arma::field<NumericVector>& log_theta, const NumericVector& eta,
    const NumericVector& sigma, const arma::field<arma::mat>& log_x_probability,
    const arma::vec& hyper_tau, const arma::vec& discrete_fields,
    int n_discrete_fields, const arma::vec& continuous_fields,
    int n_continuous_fields, int k, arma::vec n, int N);

arma::field<arma::mat> update_allocation_matrix(
    const arma::field<arma::mat>& x, const arma::mat& y,
    const arma::field<arma::mat>& z, const arma::field<IntegerVector>& lambda,
    const arma::field<arma::mat>& nu,
    const arma::field<NumericVector>& log_theta, const NumericVector& eta,
    const NumericVector& sigma, const arma::field<arma::mat>& log_x_probability,
    const arma::vec& hyper_tau, const arma::vec& discrete_fields,
    int n_discrete_fields, const arma::vec& continuous_fields,
    int n_continuous_fields, int k, arma::vec n, int N);

arma::vec calculate_discomfort_probability(
    const arma::field<arma::mat>& allocation_matrix,
    const arma::field<IntegerVector>& lambda, int k);

double f_tanh(double t, double s, double a = 1.0);
double g_tanh(double t, double s, double a = 1.0);
double f_poly(double t, double s);
double g_poly(double t, double s);

double calculate_ESS(const arma::vec& discomfort_probability,
                     double decaying_parameter, double batch_size_double);

double optimize_decaying_parameter(const arma::vec& discomfort_probability,
                                   double batch_size_double,
                                   double decaying_upper_bound);

arma::field<arma::mat> update_nu(
    const arma::field<arma::mat>& x, const arma::mat& y,
    const arma::field<arma::mat>& z, const arma::vec& discrete_fields,
    int n_discrete_fields, const arma::vec& continuous_fields,
    int n_continuous_fields, int k, const arma::vec& n, int N,
    const arma::vec& log_phi_tau, const arma::vec& hyper_delta,
    const arma::vec& hyper_tau);

#endif