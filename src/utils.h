#ifndef UTILS_H
#define UTILS_H

#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

/******************************************************************************
    Snippet:
        A collection of commonly used functions,
        including matrix manipulation and operation.
******************************************************************************/
arma::mat create_combination(int n, int start = 0);

arma::mat compile_matrix(arma::field<arma::mat> x);

arma::field<arma::mat> split_matrix(arma::mat x, arma::vec n, int k);

arma::mat create_index_matrix(int k, arma::vec n);

arma::uvec sample_index(int N, int size);

int sample_integer(arma::uvec x);

arma::field<arma::uvec> find_max_indexes(arma::field<arma::mat> probs, int k,
                                         arma::vec n);

IntegerVector get_top_n_indexes(NumericVector x, NumericVector y, int n);

arma::rowvec swap_row_values(arma::rowvec x, int i1, int i2);

arma::mat create_matrix(int n_rows, int n_cols);

arma::vec create_vector(int n_length);

arma::field<arma::mat> create_matrix_field(int n_fields, arma::vec n_rows,
                                           int n_cols);

arma::field<arma::mat> create_matrix_field(int n_fields, int n_rows,
                                           arma::vec n_cols);

arma::field<arma::vec> create_vector_field(int n_fields, arma::vec n_lengths);

String convert_pair_to_string(int i, int j);

double log_sum_exp(NumericVector x);

double log_sum_exp(arma::rowvec x);

NumericVector log_softmax(NumericVector x, bool log = true);

arma::rowvec log_softmax(arma::rowvec x, bool log = true);

double clip_ratio(double a, double b, double tol = 1e-9);

bool has_name(List x, std::string name);

arma::ivec append_vectors(arma::field<arma::ivec> vectors, int k, int N);

arma::field<arma::ivec> find_duplicates(const arma::ivec& vec);

/******************************************************************************
    Record Linkage:
        Functions commonly used when performing record linkage tasks.
******************************************************************************/

arma::mat init_true_latent(const arma::field<arma::vec>& lambda_init,
                           const arma::mat& x_append, int k, int p, int N);

arma::field<arma::mat> init_distortion(
    const arma::field<arma::mat>& x, const arma::mat& y,
    const arma::field<arma::vec>& lambda_init, int k, arma::vec n, int p,
    arma::vec discrete_fields, int n_discrete_fields,
    arma::vec continuous_fields, int n_continuous_fields, arma::vec hyper_delta,
    arma::vec hyper_tau);

arma::field<arma::ivec> identical_matching(arma::mat x, int N, arma::vec fields,
                                           bool linked_only = true);

arma::field<arma::ivec> sample_links(IntegerVector selected,
                                     arma::field<arma::ivec> identical_links);

arma::field<arma::vec> get_initial_lambda(arma::field<arma::ivec> link_sample,
                                          arma::mat index_matrix, int k,
                                          arma::vec n, int N);

arma::field<arma::vec> initial_linkage(
    arma::mat x, int k, arma::vec n, int N, arma::vec non_fuzzy_fields,
    arma::mat index_matrix, double p_init,
    Nullable<List> initial_values = R_NilValue);

arma::mat posterior_similarity(arma::field<arma::mat> probs_field,
                               bool symmetric = true);

arma::mat posterior_similarity(arma::mat samples);

/******************************************************************************
    Random Number Generator:
        Functions of random number generators that do not exist in R or Rcpp.
******************************************************************************/
int rbernoulli(double p);

NumericVector rdirichlet(NumericVector alpha);

int rcategorical(NumericVector probs, int size);

double rinvgamma(double shape, double scale);

#endif