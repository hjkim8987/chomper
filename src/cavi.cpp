#include <RcppArmadillo.h>

#include "utils.h"

// [[Rcpp::depends(RcppArmadillo)]]

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

// Initialize the average distortion rate rho_ij(n(i))
// using the prior distribution of beta_l, Beta(a(l), b(l))
//
// @param a_l double, 1st shape parameter of a Beta distribution
// @param b_l double, 2nd shape parameter of a Beta distribution
// @param n_i int, number of records in the file i
// @return initialized n_i by 1 numeric vector
arma::colvec init_rho_ij(double a_l, double b_l, int n_i) {
  arma::colvec rho_ij(n_i, arma::fill::ones);

  // Initial rho_ij will have the same average distortion rate (prior average)
  rho_ij *= a_l / (a_l + b_l);

  return rho_ij;
}

// Initialize Multinomial distribution of y_j' with parameter gamma
// using the pre-specified initial linkage index
//
// @param x original data
// @param linkage_index pre-specified linkage structure
// @param k number of files
// @param n vector of number of records in each file
// @param N total number of records, sum(n)
// @param M number of categories of each field
// @param discrete_fields indices of discrete fields
// @param n_discrete_fields number of discrete fields
// @param hyper_delta range of fuzziness
// @return initialized gamma, length p field with N by M(l) matrices
arma::field<arma::mat> init_gamma(const arma::field<arma::mat> &x,
                                  const arma::field<arma::vec> &linkage_index,
                                  int k, arma::vec n, int N, arma::vec M,
                                  arma::vec discrete_fields,
                                  int n_discrete_fields,
                                  arma::vec hyper_delta) {
  arma::field<arma::mat> gamma(n_discrete_fields);

  // Multiplier for the probability of the initial linkages
  // i.e., nu = (1, 1, 1, mult, ..., 1, 1, 1)
  double mult = 10.0;

  for (int li = 0; li < n_discrete_fields; li++) {
    int l = discrete_fields[li];
    int delta = hyper_delta[li];

    arma::mat gamma_l(N, M(li), arma::fill::ones);

    IntegerVector appeared;
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n[i]; j++) {
        IntegerVector jprime = {int(linkage_index(i)(j))};
        // IntegerVector jprime[1];
        // jprime[0] = linkage_index(i)(j);
        if (!in(jprime, appeared)(0)) {
          // if current linkage index, j', is not appeared yet,
          // and if the value of x_ijl is in the range of fuzziness,
          // assign high probability mass to the distribution
          for (int m = 1; m <= M[li]; m++) {
            gamma_l(jprime[0], m - 1) *=
                pow(mult, (int)(std::fabs(x(i)(j, l) - m) <= delta));
          }
          // append jprime
          appeared.push_back(jprime(0));
        }
      }
    }
    // normalize the probabilities
    // It should be done out of the loop
    // because there may be unassigned records
    for (int jp = 0; jp < N; jp++) {
      gamma_l.row(jp) /= sum(gamma_l.row(jp));
    }
    gamma(li) = gamma_l;
  }

  return gamma;
}

// Initialize the Mean of Normal distribution of y_j'
// using the pre-specified initial linkage index
//
// @param x original data
// @param linkage_index pre-specified linkage structure
// @param k number of files
// @param n vector of number of records in each file
// @param N total number of records, sum(n)
// @param N_double double, total number of records, sum(n)
// @param continuous_fields indices of continuous fields
// @param n_continuous_fields number of continuous fields
// @return initialized eta_tilde, n_continuous_fields by N matrix
arma::mat init_y_mean(const arma::field<arma::mat> &x,
                      const arma::field<arma::vec> &linkage_index, int k,
                      arma::vec n, int N, double N_double,
                      arma::vec continuous_fields, int n_continuous_fields) {
  arma::mat eta_tilde(n_continuous_fields, N, arma::fill::zeros);

  // Initialize eta_tilde with the assigned value, x_ijl
  IntegerVector visited;
  arma::mat x_sum(1, n_continuous_fields, arma::fill::zeros);
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n[i]; j++) {
      for (int li = 0; li < n_continuous_fields; li++) {
        int l = continuous_fields[li];
        visited.push_back(linkage_index(i)(j));
        eta_tilde(li, linkage_index(i)(j)) = x(i)(j, l);
        x_sum(0, li) += x(i)(j, l);
      }
    }
  }

  // For the unassigned j', assign the mean of the data
  x_sum /= N_double;
  for (int jprime = 0; jprime < N; jprime++) {
    IntegerVector jprime_vec = {jprime};
    if (!in(jprime_vec, visited)(0)) {
      for (int li = 0; li < n_continuous_fields; li++) {
        eta_tilde(li, jprime) = x_sum(0, li);
      }
    }
  }

  return eta_tilde;
}

// Initialize the Variance of Normal distribution of y_j'
// using the sample variance of the data;
// because the initial linkage structure is set with
// identically matching records, which makes the variance be 0.
//
// @param x original data
// @param k number of files
// @param n vector of number of records in each file
// @param N total number of records, sum(n)
// @param N_double double, total number of records, sum(n)
// @param continuous_fields indices of continuous fields
// @param n_continuous_fields number of continuous fields
// @return initialized sigma_tilde, n_continuous_fields by N matrix
arma::mat init_y_var(const arma::field<arma::mat> &x, int k, arma::vec n, int N,
                     double N_double, arma::vec continuous_fields,
                     int n_continuous_fields) {
  arma::mat sigma_tilde(n_continuous_fields, N, arma::fill::ones);

  for (int li = 0; li < n_continuous_fields; li++) {
    int l = continuous_fields[li];

    double summation = 0.0;
    for (int i = 0; i < k; i++) {
      summation += sum(x(i).col(l));
    }
    double sample_mean = summation / N_double;

    double squared_summation = 0.0;
    for (int i = 0; i < k; i++) {
      squared_summation += sum(pow(x(i).col(l) - sample_mean, 2.0));
    }

    double sample_var = squared_summation / (N_double - 1.0);

    for (int jprime = 0; jprime < N; jprime++) {
      sigma_tilde(li, jprime) = sample_var;
    }
  }

  return sigma_tilde;
}

// Initialize Inverse-Gamma scale parameter for sigma
//
// @param hyper_sigma Inverse-Gamma parameter for variance of latent true values
// @param sigma_shape shape parameter of Inverse-Gamma distribution
// @param eta_tilde mean of y_j'
// @param sigma_tilde variance of y_j'
// @param N_double double, total number of records, sum(n)
// @param n_continuous_fields number of continuous fields
// @return initialized sigma_scale, length n_continuous_fields vector
arma::vec init_sigma_scale(arma::mat hyper_sigma, arma::vec sigma_shape,
                           arma::mat eta_tilde, arma::mat sigma_tilde,
                           double N_double, int n_continuous_fields) {
  arma::vec sigma_scale(n_continuous_fields, arma::fill::ones);

  for (int li = 0; li < n_continuous_fields; li++) {
    sigma_scale(li) =
        hyper_sigma(li, 1) +
        0.5 * sum(sigma_tilde.col(li) + pow(eta_tilde.col(li), 2) -
                  2.0 * eta_tilde.col(li) * mean(eta_tilde.col(li)) +
                  sigma_shape(li) / N_double + pow(mean(eta_tilde.col(li)), 2));
  }

  return sigma_scale;
}

// Initialize Multinomial distribution of lambda_ij with parameter nu
// using the pre-specified initial index
//
// @param index pre-specified linkage structure
// @param k number of files
// @param n vector of number of records in each file
// @param N total number of records, sum(n)
// @return initialized nu, length k field with N by 1 vector
arma::field<arma::mat> init_nu(arma::field<arma::vec> index, int k, arma::vec n,
                               int N) {
  arma::field<arma::mat> nu(k);

  // Multiplier for the probability of the latent level
  // i.e., gamma = (1, 1, 1, mult, ..., 1, 1, 1)
  double mult = 10.0;

  for (int i = 0; i < k; i++) {
    arma::mat nu_i(n(i), N, arma::fill::ones);
    for (int j = 0; j < n(i); j++) {
      nu_i(j, index(i)(j)) *= mult;
      nu_i.row(j) /= sum(nu_i.row(j));
    }
    nu(i) = nu_i;
  }

  return nu;
}

/*******************************************
      1. Initializers
    2. Utility Functions
      3. Updaters
      4. ELBO Calculator
*******************************************/

// Sum rho_ijl for specific l, i.e., sum_ij rho_ijl
// It is used for either optimization or calculating ELBO
//
// @param rho average distortion rate of x_ijl
// @param k number of files
// @param N_double double, total number of records, sum(n)
// @param p number of common fields
// @param complement bool, if true, it calculates sum of (1 - rho_ijl)
// @return result of summation, p by 1 numeric vector
arma::vec sum_ij_rho_ijl(const arma::field<arma::mat> &rho, int k,
                         double N_double, int p, bool complement) {
  arma::vec sum_ij_rho_ij(p, arma::fill::ones);
  for (int l = 0; l < p; l++) {
    for (int i = 0; i < k; i++) {
      if (complement) {
        sum_ij_rho_ij(l) += (N_double - sum(rho(i).col(l)));
      } else {
        sum_ij_rho_ij(l) += sum(rho(i).col(l));
      }
    }
  }
  return sum_ij_rho_ij;
}

// Calculate log[Gamma(sum_i x_i) / prod_i Gamma(x_i)]
//
// @param x n by 1 numeric vector
// @param n length of x
// @return result double
double log_beta_constant(arma::vec x, int n) {
  double result = R::lgammafn(sum(x));
  for (int i = 0; i < n; i++) {
    result -= R::lgammafn(x(i));
  }
  return result;
}

// Calculate beta function
// R::beta(a, b) should be avoided
// because it returns 0 when a and b are large.
//
// @param a first parameter
// @param b second parameter
// @param log bool, if true, return log of beta function
// @return result double
double beta_function(double a, double b, bool log) {
  if (log) {
    return R::lgammafn(a) + R::lgammafn(b) - R::lgammafn(a + b);
  } else {
    return exp(R::lgammafn(a) + R::lgammafn(b) - R::lgammafn(a + b));
  }
}

/*******************************************
      1. Initializers
      2. Utility Functions
    3. Updaters
      4. ELBO Calculator
*******************************************/

// Update Dirichlet parameters for theta
//
// @param x original data
// @param mu set of M(l) by 1 hyperparameter vectors for theta_l
// @param gamma multinomial probabilities of y
// @param rho average distortion rate of x_ijl
// @param k number of files
// @param discrete_fields indices of discrete fields
// @param n_discrete_fields number of discrete fields
// @param M number of categories of each field
// @return updated parameter vectors
arma::field<arma::vec> update_alpha(const arma::field<arma::mat> &x,
                                    const arma::field<arma::vec> &mu,
                                    const arma::field<arma::mat> &gamma,
                                    const arma::field<arma::mat> &rho, int k,
                                    arma::vec discrete_fields,
                                    int n_discrete_fields, arma::vec M) {
  arma::field<arma::vec> alpha(n_discrete_fields);

  for (int li = 0; li < n_discrete_fields; li++) {
    int l = discrete_fields[li];

    arma::vec alpha_empty(M(li), arma::fill::zeros);

    for (int m = 0; m < M(li); m++) {
      double rho_sum = 0.0;
      for (int i = 0; i < k; i++) {
        rho_sum += arma::dot(x(i).col(l) == int(m + 1), rho(i).col(l));
      }
      alpha_empty(m) = mu(li)(m) + sum(gamma(li).col(m)) + rho_sum;
    }
    alpha(li) = alpha_empty;
  }

  return alpha;
}

// Update the mean of log Dirichlet:
// E[log theta_lm] = digamma(alpha_lm) - digamma(sum(alpha_lm))
// This is an auxiliary matrix used for both optimization and calculating ELBO
//
// @param alpha updated Dirichlet parameters for theta
// @param n_discrete_fields number of discrete fields
// @param M number of categories of each field
// @return psi updated expectations
arma::field<arma::vec> update_psi(const arma::field<arma::vec> &alpha,
                                  int n_discrete_fields, arma::vec M) {
  arma::field<arma::vec> psi(n_discrete_fields);

  for (int li = 0; li < n_discrete_fields; li++) {
    arma::vec psi_empty(M(li), arma::fill::zeros);

    double alpha_sum = sum(alpha(li));

    for (int m = 0; m < M(li); m++) {
      psi_empty(m) = R::digamma(alpha(li)(m)) - R::digamma(alpha_sum);
    }
    psi(li) = psi_empty;
  }

  return psi;
}

// Update mean of Normal distribution of eta
//
// @param x original data
// @param rho average distortion rate of x_ijl
// @param eta_tilde mean of Normal distribution of y_j'
// @param continuous_fields indices of continuous fields
// @param n_continuous_fields number of continuous fields
// @param rho_sum sum_ij rho_ij and N - sum_ij rho_ij
// @param k number of files
// @param N_double double, total number of records, sum(n)
// @return updated eta_mean, length n_continuous_fields vector
arma::vec update_eta_mean(const arma::field<arma::mat> &x,
                          const arma::field<arma::mat> &rho,
                          arma::mat eta_tilde, arma::vec continuous_fields,
                          int n_continuous_fields, arma::mat rho_sum, int k,
                          double N_double) {
  arma::vec eta_mean(n_continuous_fields, arma::fill::zeros);

  for (int li = 0; li < n_continuous_fields; li++) {
    int l = continuous_fields[li];

    double x_rho_sum = 0.0;
    for (int i = 0; i < k; i++) {
      x_rho_sum += arma::dot(x(i).col(l), rho(i).col(l));
    }

    eta_mean(li) =
        (sum(eta_tilde.row(li)) + x_rho_sum) / N_double + rho_sum(l, 0);
  }

  return eta_mean;
}

// Update variance of Normal distribution of eta
//
// @param sigma_shape shape parameter of Inverse-Gamma distribution
// @param sigma_scale scale parameter of Inverse-Gamma distribution
// @param continuous_fields indices of continuous fields
// @param n_continuous_fields number of continuous fields
// @param rho_sum sum_ij rho_ij and N - sum_ij rho_ij
// @param N_double double, total number of records, sum(n)
// @return updated eta_var, length n_continuous_fields vector
arma::vec update_eta_var(arma::vec sigma_shape, arma::vec sigma_scale,
                         arma::vec continuous_fields, int n_continuous_fields,
                         arma::mat rho_sum, double N_double) {
  arma::vec eta_var(n_continuous_fields, arma::fill::ones);

  for (int li = 0; li < n_continuous_fields; li++) {
    int l = continuous_fields[li];

    double N_rho_sum = N_double + rho_sum(l, 0);

    eta_var(li) =
        sigma_scale(li) / (N_rho_sum * (sigma_shape(li) + 0.5 * N_rho_sum));
  }

  return eta_var;
}

// Update Inverse-Gamma shape parameter for sigma
//
// @param hyper_sigma Inverse-Gamma parameter for variance of latent true values
// @param rho_sum sum_ij rho_ij and N - sum_ij rho_ij
// @param continuous_fields indices of continuous fields
// @param n_continuous_fields number of continuous fields
// @param N_double double, total number of records, sum(n)
// @return updated sigma_shape, length n_continuous_fields vector
arma::vec update_sigma_shape(arma::mat hyper_sigma, arma::mat rho_sum,
                             arma::vec continuous_fields,
                             int n_continuous_fields, double N_double) {
  arma::vec sigma_shape(n_continuous_fields, arma::fill::ones);

  for (int li = 0; li < n_continuous_fields; li++) {
    int l = continuous_fields[li];

    sigma_shape(li) = hyper_sigma(li, 0) + 0.5 * (N_double + rho_sum(l, 0));
  }

  return sigma_shape;
}

// Update Inverse-Gamma scale parameter for sigma
//
// @param hyper_sigma Inverse-Gamma parameter for variance of latent true values
// @param eta_tilde mean of Normal distribution of y_j'
// @param sigma_tilde variance of Normal distribution of y_j'
// @param x original data
// @param rho average distortion rate of x_ijl
// @param zeta auxiliary variable of E[eta_l] and E[eta_l^2]
// @param continuous_fields indices of continuous fields
// @param n_continuous_fields number of continuous fields
// @param k number of files
// @return updated sigma_scale, length n_continuous_fields vector
arma::vec update_sigma_scale(arma::mat hyper_sigma, arma::mat eta_tilde,
                             arma::mat sigma_tilde,
                             const arma::field<arma::mat> &x,
                             const arma::field<arma::mat> &rho, arma::mat zeta,
                             arma::vec continuous_fields,
                             int n_continuous_fields, int k) {
  arma::vec sigma_scale(n_continuous_fields, arma::fill::ones);

  for (int li = 0; li < n_continuous_fields; li++) {
    int l = continuous_fields[li];

    double rho_x_sum = 0.0;
    for (int i = 0; i < k; i++) {
      rho_x_sum += arma::dot(rho(i).col(l),
                             pow(x(i).col(l), 2.0) -
                                 2.0 * x(i).col(l) * zeta(li, 0) + zeta(li, 1));
    }

    sigma_scale(li) =
        hyper_sigma(li, 1) +
        0.5 * (sum(sigma_tilde.row(li) + pow(eta_tilde.row(li), 2.0) -
                   2.0 * eta_tilde.row(li) * zeta(li, 0) + zeta(li, 1)) +
               rho_x_sum);
  }

  return sigma_scale;
}

// Update Beta parameters for beta
//
// @param hyper_beta Beta hyper-parameters for distortion rate
// @param rho_sum sum_ij rho_ij and N - sum_ij rho_ij
// @param p total number of fields
// @return updated parameters
arma::mat update_omega(arma::mat hyper_beta, arma::mat rho_sum, int p) {
  arma::mat omega(p, 2, arma::fill::ones);

  for (int l = 0; l < p; l++) {
    omega(l, 0) = hyper_beta(l, 0) + rho_sum(l, 0);
    omega(l, 1) = hyper_beta(l, 1) + rho_sum(l, 1);
  }

  return omega;
}

// Update the mean of log Beta, E[log beta_l]
// This is an auxiliary matrix used for both optimization and calculating ELBO
//
// @param omega p by 2 matrix of updated parameters of beta
// @param p number of common fields
// @return kappa updated expectations
arma::mat update_kappa(arma::mat omega, int p) {
  arma::mat kappa(p, 2, arma::fill::zeros);

  for (int l = 0; l < p; l++) {
    double digam_all = R::digamma(sum(omega.row(l)));

    kappa(l, 0) = R::digamma(omega(l, 0)) - digam_all;
    kappa(l, 1) = R::digamma(omega(l, 1)) - digam_all;
  }

  return kappa;
}

// Update Multinomial distribution of lambda_ij with parameter nu
//
// @param x original data
// @param rho average distortion rate of x_ijl
// @param gamma multinomial probabilities of y_j'l
// @param eta_tilde mean of Normal distribution of y_j'l
// @param sigma_tilde variance of Normal distribution of y_j'l
// @param k number of files
// @param n vector of number of records in each file
// @param N total number of records, sum(n)
// @param hyper_tau temperature parameter
// @param log_phi_tau log(phi^(1 / tau))
// @param discrete_fields indices of discrete fields
// @param n_discrete_fields number of discrete fields
// @param M number of categories of each field
// @param hyper_delta fuzzy range
// @param continuous_fields indices of continuous fields
// @param n_continuous_fields number of continuous fields
// @return updated nu and log_nu, length k field with n(i) by N matrices
arma::field<arma::field<arma::mat>> update_nu_all(
    const arma::field<arma::mat> &x, const arma::field<arma::mat> &rho,
    const arma::field<arma::mat> &gamma, arma::mat eta_tilde,
    arma::mat sigma_tilde, int k, arma::vec n, int N, arma::vec hyper_tau,
    arma::vec log_phi_tau, arma::vec discrete_fields, int n_discrete_fields,
    arma::vec M, arma::vec hyper_delta, arma::vec continuous_fields,
    int n_continuous_fields) {
  arma::field<arma::field<arma::mat>> nu_all(2);

  arma::field<arma::mat> nu(k);
  arma::field<arma::mat> log_nu(k);

  for (int i = 0; i < k; i++) {
    arma::mat nu_i(n(i), N, arma::fill::ones);
    arma::mat log_nu_i(n(i), N, arma::fill::ones);

    for (int j = 0; j < n(i); j++) {
      arma::rowvec continuous_minus_discrete(N, arma::fill::zeros);

      for (int jprime = 0; jprime < N; jprime++) {
        double discrete_sum = 0.0;
        for (int li = 0; li < n_discrete_fields; li++) {
          int l = discrete_fields[li];

          double gamma_sum = 0.0;
          for (int m = 1; m <= M[li]; m++) {
            gamma_sum += gamma(li)(jprime, m - 1) *
                         (double)(std::fabs(x(i)(j, l) - m) <= hyper_delta[li]);
          }
          discrete_sum += (1.0 - rho(i)(j, l)) * gamma_sum * log_phi_tau(li);
        }

        double continuous_sum = 0.0;
        arma::colvec expectation_jprime = eta_tilde.col(jprime);
        arma::colvec squared_expectation_jprime =
            sigma_tilde.col(jprime) + pow(expectation_jprime, 2.0);
        for (int li = 0; li < n_continuous_fields; li++) {
          int l = continuous_fields[li];

          continuous_sum += 0.5 * (1.0 - rho(i)(j, l)) *
                            (pow(x(i)(j, l), 2.0) -
                             2.0 * x(i)(j, l) * expectation_jprime(li) +
                             squared_expectation_jprime(li)) /
                            hyper_tau(l);
        }
        // nu(i)(j, jprime) = exp(discrete_sum - continuous_sum);
        continuous_minus_discrete(jprime) = discrete_sum - continuous_sum;
      }
      // normalize the probabilities
      // nu(i).row(j) = exp(probs) / sum(exp(probs));
      arma::rowvec probs = log_softmax(continuous_minus_discrete, false);
      arma::rowvec log_probs = log_softmax(continuous_minus_discrete, true);

      nu_i.row(j) = probs;
      log_nu_i.row(j) = log_probs;
    }
    nu(i) = nu_i;
    log_nu(i) = log_nu_i;
  }

  nu_all(0) = nu;
  nu_all(1) = log_nu;

  return nu_all;
}

// Update Multinomial distribution of y_j'l with parameter gamma
// This function is used for y_j'l for discrete fields
//
// @param x original data
// @param rho average distortion rate of x_ijl
// @param nu Multinomial probabilities of lambda_ij
// @param psi mean of log Dirichlet, E[log theta_lm]
// @param k number of files
// @param n vector of number of records in each file
// @param N total number of records, sum(n)
// @param discrete_fields indices of discrete fields
// @param n_discrete_fields number of discrete fields
// @param M number of categories of each field
// @param hyper_tau temperature parameter
// @param hyper_delta fuzzy range
// @param log_phi_tau log(phi^(1 / tau))
// @return updated gamma and log_gamma,
//         length n_discrete_fields field with N by M(l) matrices
arma::field<arma::field<arma::mat>> update_gamma(
    const arma::field<arma::mat> &x, const arma::field<arma::mat> &rho,
    const arma::field<arma::mat> &nu, const arma::field<arma::vec> &psi, int k,
    arma::vec n, int N, arma::vec discrete_fields, int n_discrete_fields,
    arma::vec M, arma::vec hyper_tau, arma::vec hyper_delta,
    arma::vec log_phi_tau) {
  arma::field<arma::field<arma::mat>> gamma_all(2);

  arma::field<arma::mat> gamma(n_discrete_fields);
  arma::field<arma::mat> log_gamma(n_discrete_fields);
  for (int li = 0; li < n_discrete_fields; li++) {
    int l = discrete_fields[li];
    int delta = hyper_delta[li];

    arma::mat gamma_empty(N, M(li), arma::fill::zeros);
    arma::mat log_gamma_empty(N, M(li), arma::fill::zeros);
    for (int jprime = 0; jprime < N; jprime++) {
      for (int m = 0; m < M(li); m++) {
        double rho_sum = 0.0;
        for (int i = 0; i < k; i++) {
          arma::colvec x_il(n(i), arma::fill::zeros);
          for (int j = 0; j < n(i); j++) {
            x_il(j) = (std::fabs(x(i)(j, l) - (m + 1)) <= delta) *
                      (1.0 - rho(i)(j, l));
          }

          rho_sum += arma::dot(nu(i).col(jprime), x_il);
        }
        gamma_empty(jprime, m) = rho_sum * log_phi_tau(li) + psi(li)(m);
      }

      // normalize the probabilities
      gamma_empty.row(jprime) = log_softmax(gamma_empty.row(jprime), false);
      log_gamma_empty.row(jprime) = log_softmax(gamma_empty.row(jprime), true);
    }
    gamma(li) = gamma_empty;
    log_gamma(li) = log_gamma_empty;
  }

  gamma_all(0) = gamma;
  gamma_all(1) = log_gamma;

  return gamma_all;
}

// Update chi = gamma * nu
// It is an auxiliary variable used for updating rho
//
// @param x original data
// @param gamma multinomial probabilities of y_j'l
// @param nu Multinomial probabilities of lambda_ij
// @param k number of files
// @param n vector of number of records in each file
// @param M number of categories of each field
// @param N total number of records, sum(n)
// @param discrete_fields indices of discrete fields
// @param n_discrete_fields number of discrete fields
// @param hyper_delta fuzzy range
// @return updated chi, length k field with n(i) by n_discrete_fields matrix
arma::field<arma::mat> update_chi_discrete(
    const arma::field<arma::mat> &x, const arma::field<arma::mat> &gamma,
    const arma::field<arma::mat> &nu, int k, arma::vec n, arma::vec M, int N,
    arma::vec discrete_fields, int n_discrete_fields, arma::vec hyper_delta) {
  arma::field<arma::mat> chi(k);

  for (int i = 0; i < k; i++) {
    arma::mat chi_empty(n(i), n_discrete_fields, arma::fill::zeros);
    chi(i) = chi_empty;

    for (int j = 0; j < n(i); j++) {
      for (int li = 0; li < n_discrete_fields; li++) {
        int l = discrete_fields[li];
        int delta = hyper_delta[li];

        double x_ijl = x(i)(j, l);

        double chi_sum = 0.0;
        for (int jprime = 0; jprime < N; jprime++) {
          double gamma_sum = 0.0;
          for (int m = 1; m <= M(li); m++) {
            gamma_sum += gamma(li)(jprime, m - 1) *
                         (double)(std::fabs(x_ijl - m) <= delta);
          }

          chi_sum += gamma_sum * nu(i)(j, jprime);
        }

        chi(i)(j, li) = chi_sum;
      }
    }
  }

  return chi;
}

// Update parameters of Normal distribution of y_j'l
// Calculate the mean and variance of y_j'l simultaneously,
// because they share some values
//
// @param x original data
// @param rho average distortion rate of x_ijl
// @param nu Multinomial probabilities of lambda_ij
// @param eta_mean mean of Normal distribution of eta
// @param sigma_shape shape parameter of Inverse-Gamma distribution (sigma)
// @param sigma_scale scale parameter of Inverse-Gamma distribution (sigma)
// @param hyper_tau temperature parameter
// @param k number of files
// @param n vector of number of records in each file
// @param N total number of records, sum(n)
// @param continuous_fields indices of continuous fields
// @param n_continuous_fields number of continuous fields
// @return updated parameters, field with n_continuous_fields by N matrix
arma::field<arma::mat> update_continuous_y(
    const arma::field<arma::mat> &x, const arma::field<arma::mat> &rho,
    const arma::field<arma::mat> &nu, arma::vec eta_mean, arma::vec sigma_shape,
    arma::vec sigma_scale, arma::vec hyper_tau, int k, arma::vec n, int N,
    arma::vec continuous_fields, int n_continuous_fields) {
  arma::mat eta_tilde(n_continuous_fields, N, arma::fill::zeros);
  arma::mat sigma_tilde(n_continuous_fields, N, arma::fill::zeros);

  for (int li = 0; li < n_continuous_fields; li++) {
    int l = continuous_fields[li];

    double eta_mean_l = eta_mean(li);
    double sigma_inverse_mean_l = sigma_shape(li) / sigma_scale(li);
    double one_over_tau_l = 1.0 / hyper_tau(l);

    for (int jprime = 0; jprime < N; jprime++) {
      double x_summand = 0.0;
      double summand = 0.0;
      for (int i = 0; i < k; i++) {
        for (int j = 0; j < n(i); j++) {
          double inner_summand = nu(i)(j, jprime) * (1.0 - rho(i)(j, l));
          x_summand += x(i)(j, l) * inner_summand;
          summand += inner_summand;
        }
      }
      sigma_tilde(li, jprime) =
          1.0 / (sigma_inverse_mean_l + one_over_tau_l * summand);
      eta_tilde(li, jprime) =
          sigma_tilde(li, jprime) *
          (eta_mean_l * sigma_inverse_mean_l + one_over_tau_l * x_summand);
    }
  }

  arma::field<arma::mat> y_params(2);
  y_params(0) = eta_tilde;
  y_params(1) = sigma_tilde;

  return y_params;
}

// Update chi_expectation = nu * E[y_j'l]
// It is an auxiliary variable used for updating rho
//
// @param nu Multinomial probabilities of lambda_ij
// @param eta_tilde mean of Normal distribution of y_j'l
// @param k number of files
// @param n vector of number of records in each file
// @param n_continuous_fields number of continuous fields
// @return updated chi, length k field with n(i) by n_continuous_fields matrix
arma::field<arma::mat> update_chi_expectation(const arma::field<arma::mat> &nu,
                                              arma::mat eta_tilde, int k,
                                              arma::vec n,
                                              int n_continuous_fields) {
  arma::field<arma::mat> chi(k);

  for (int i = 0; i < k; i++) {
    arma::mat chi_empty(n(i), n_continuous_fields, arma::fill::zeros);
    chi(i) = chi_empty;

    for (int j = 0; j < n(i); j++) {
      arma::rowvec nu_ij = nu(i).row(j);
      for (int li = 0; li < n_continuous_fields; li++) {
        chi(i)(j, li) = arma::dot(nu_ij, eta_tilde.row(li));
      }
    }
  }

  return chi;
}

// Update chi_squared_expectation = nu * E[y_j'l^2]
// It is an auxiliary variable used for updating rho
//
// @param nu Multinomial probabilities of lambda_ij
// @param eta_tilde mean of Normal distribution of y_j'l
// @param sigma_tilde variance of Normal distribution of y_j'l
// @param k number of files
// @param n vector of number of records in each file
// @param n_continuous_fields number of continuous fields
// @return updated chi, length k field with n(i) by n_continuous_fields matrix
arma::field<arma::mat> update_chi_squared_expectation(
    const arma::field<arma::mat> &nu, arma::mat eta_tilde,
    arma::mat sigma_tilde, int k, arma::vec n, int n_continuous_fields) {
  arma::field<arma::mat> chi(k);
  for (int i = 0; i < k; i++) {
    arma::mat chi_empty(n(i), n_continuous_fields, arma::fill::zeros);
    chi(i) = chi_empty;

    for (int j = 0; j < n(i); j++) {
      arma::rowvec nu_ij = nu(i).row(j);
      for (int li = 0; li < n_continuous_fields; li++) {
        chi(i)(j, li) =
            arma::dot(nu_ij, sigma_tilde.row(li) + pow(eta_tilde.row(li), 2.0));
      }
    }
  }
  return chi;
}

// Update zeta
// It is an auxiliary variable used for updating rho and sigma
// zeta = n_continuous_fields by 2 matrix with
//        rows of (E[eta_l], E[eta_l^2])
// It should be updated twice:
//  1. Before updating sigma
//  2. Before updating rho
//
// @param x original data
// @param rho average distortion rate of x_ijl
// @param eta_tilde mean of Normal distribution of y_j'l
// @param eta_var variance of Normal distribution of eta_l
// @param rho_sum sum_ij E[z_ijl], sum_ij E[1 - z_ijl]
// @param continuous_fields indices of continuous fields
// @param n_continuous_fields number of continuous fields
// @param k number of files
// @param N_double total number of records, sum(n)
// @return updated zeta, n_continuous_fields by 2 matrix
arma::mat update_zeta(const arma::field<arma::mat> &x,
                      const arma::field<arma::mat> &rho, arma::mat eta_tilde,
                      arma::vec eta_var, arma::vec sigma_shape,
                      arma::vec sigma_scale, arma::mat rho_sum,
                      arma::vec continuous_fields, int n_continuous_fields,
                      int k, double N_double, bool before_sigma) {
  arma::mat zeta(n_continuous_fields, 2, arma::fill::zeros);

  for (int li = 0; li < n_continuous_fields; li++) {
    int l = continuous_fields[li];

    double numerator = sum(eta_tilde.row(li));
    for (int i = 0; i < k; i++) {
      numerator += arma::dot(x(i).col(l), rho(i).col(l));
    }
    double denominator = N_double + rho_sum(l, 0);

    zeta(li, 0) = numerator / denominator;
    zeta(li, 1) = pow(zeta(li, 0), 2.0);

    if (before_sigma) {
      zeta(li, 1) += eta_var(li);
    } else {
      zeta(li, 1) += sigma_scale(li) / (denominator * sigma_shape(li));
    }
  }

  return zeta;
}

// Update parameter of Bernoulli distribution of z_ijl
//
// @param x original data
// @param k number of files
// @param n vector of number of records in each file
// @param p number of fields
// @param discrete_fields indices of discrete fields
// @param n_discrete_fields number of discrete fields
// @param M number of categories of each field
// @param log_constant_matrix log(constants) for normalizing softmax
// @param sigma_shape shape parameter of Inverse-Gamma distribution (sigma)
// @param sigma_scale scale parameter of Inverse-Gamma distribution (sigma)
// @param psi mean of log Dirichlet, E[log theta_lm]
// @param chi_discrete chi = gamma * nu
// @param log_phi_tau log(phi^(1 / tau))
// @param continuous_fields indices of continuous fields
// @param n_continuous_fields number of continuous fields
// @param chi_expectation E[y_j'l]
// @param chi_squared_expectation E[y_j'l^2]
// @param zeta E[eta_l], E[eta_l^2]
// @param hyper_tau temperature parameter
// @param kappa the mean of log Beta, E[log beta_l]
// @return updated rho, length k field with n(i) by p matrix
arma::field<arma::mat> update_rho(
    const arma::field<arma::mat> &x, int k, arma::vec n, int p,
    arma::vec discrete_fields, int n_discrete_fields, arma::vec M,
    const arma::field<arma::mat> &log_constant_matrix, arma::vec sigma_shape,
    arma::vec sigma_scale, const arma::field<arma::vec> &psi,
    const arma::field<arma::mat> &chi_discrete, arma::vec log_phi_tau,
    arma::vec continuous_fields, int n_continuous_fields,
    const arma::field<arma::mat> &chi_expectation,
    const arma::field<arma::mat> &chi_squared_expectation, arma::mat zeta,
    arma::vec hyper_tau, arma::mat kappa) {
  arma::field<arma::mat> rho(k);

  double log_2pi = log(2.0 * M_PI);

  for (int i = 0; i < k; i++) {
    arma::mat rho_empty(n(i), p, arma::fill::zeros);
    rho(i) = rho_empty;

    for (int j = 0; j < n(i); j++) {
      for (int li = 0; li < n_discrete_fields; li++) {
        int l = discrete_fields[li];

        double psi_sum = 0.0;
        for (int m = 0; m < M(li); m++) {
          psi_sum += (double)(x(i)(j, l) == int(m + 1)) * psi(li)(m);
        }

        double numerator = exp(kappa(l, 0) + psi_sum);
        double denominator =
            numerator +
            exp(kappa(l, 1) + chi_discrete(i)(j, li) * log_phi_tau(li) -
                log_constant_matrix(i)(j, li));

        rho(i)(j, l) = clip_ratio(numerator, denominator);
      }

      for (int li = 0; li < n_continuous_fields; li++) {
        int l = continuous_fields[li];

        double numerator =
            exp(kappa(l, 0) -
                0.5 * (log_2pi + log(sigma_scale(li)) -
                       R::digamma(sigma_shape(li)) +
                       sigma_shape(li) *
                           (pow(x(i)(j, l), 2.0) -
                            2.0 * x(i)(j, l) * zeta(li, 0) + zeta(li, 1)) /
                           sigma_scale(li)));

        double denominator =
            numerator +
            exp(kappa(l, 1) -
                0.5 * (log_2pi + log(hyper_tau(l)) +
                       (pow(x(i)(j, l), 2.0) -
                        2.0 * x(i)(j, l) * chi_expectation(i)(j, li) +
                        chi_squared_expectation(i)(j, li)) /
                           hyper_tau(l)));

        rho(i)(j, l) = clip_ratio(numerator, denominator);
      }
    }
  }

  return rho;
}

/*******************************************
      1. Initializers
      2. Utility Functions
      3. Updaters
    4. ELBO Calculator
*******************************************/

// Calculate ELBO(q) for the current iteration
// It uses the original data x and the fully updated parameters
//
// @param x original data
// @param rho q_z ~ Bernoulli(rho)
// @param rho_sum sum_ij E[z_ijl], sum_ij E[1 - z_ijl]
// @param nu q_lambda ~ MN(1, nu)
// @param log_nu log(nu)
// @param gamma q_y ~ MN(1, gamma) (discrete fields)
// @param log_gamma log(gamma) (discrete fields)
// @param alpha q_theta ~ Dirichlet(alpha)
// @param psi E[log theta_l]
// @param omega q_beta ~ Beta(omega[l, 0], omega[l, 1])
// @param eta_tilde q_y ~ N(eta_tilde[l, j'], sigma_tilde[l, j']) (continuous)
// @param sigma_tilde q_y ~ N(eta_tilde[l, j'], sigma_tilde[l, j']) (continuous)
// @param sigma_shape shape parameter of Inverse-Gamma distribution (sigma)
// @param sigma_scale scale parameter of Inverse-Gamma distribution (sigma)
// @param chi_discrete chi = gamma * nu
// @param chi_expectation nu * E[y_j'l]
// @param chi_squared_expectation nu * E[y_j'l^2]
// @param hyper_mu Dirichlet for the multinomial theta
// @param hyper_tau temperature parameter
// @param log_phi_tau log(phi^(1 / tau))
// @param log_constant_matrix log(constants) for normalizing softmax
// @param k number of files
// @param n vector of number of records in each file
// @param N total number of records, sum(n)
// @param N_double N as double
// @param discrete_fields indices of discrete fields
// @param n_discrete_fields number of discrete fields
// @param M number of categories of each field
// @param continuous_fields indices of continuous fields
// @param n_continuous_fields number of continuous fields
// @return elbo double, ELBO for the new iteration
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
    arma::vec M, arma::vec continuous_fields, int n_continuous_fields) {
  double elbo = 0.0;

  double log_2pi = log(2.0 * M_PI);

  double temp_sum = 0.0;
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n(i); j++) {
      for (int li = 0; li < n_discrete_fields; li++) {
        int l = discrete_fields[li];

        elbo +=
            rho(i)(j, l) * (psi(li)((int)(x(i)(j, l) - 1)) - log(rho(i)(j, l)));
        elbo += (1.0 - rho(i)(j, l)) *
                (chi_discrete(i)(j, li) * log_phi_tau(li) -
                 log_constant_matrix(i)(j, li) - log(1.0 - rho(i)(j, l)));
      }
      elbo -= arma::dot(nu(i).row(j), log_nu(i).row(j));

      for (int li = 0; li < n_continuous_fields; li++) {
        int l = continuous_fields[li];

        elbo -= rho(i)(j, l) * log(rho(i)(j, l)) +
                (1.0 - rho(i)(j, l)) * log(1.0 - rho(i)(j, l));

        temp_sum -= rho(i)(j, l) * (log_2pi + log(sigma_scale(li)) +
                                    sigma_shape(li) *
                                        (pow(x(i)(j, l), 2.0) -
                                         2.0 * x(i)(j, l) * zeta(li, 0) +
                                         pow(zeta(li, 0), 2.0)) /
                                        sigma_scale(li) +
                                    1.0 / (N_double + rho_sum(l, 0)));
        temp_sum -= (1.0 - rho(i)(j, l)) *
                    (log_2pi + log(hyper_tau(l)) +
                     (pow(x(i)(j, l), 2.0) -
                      2.0 * x(i)(j, l) * chi_expectation(i)(j, li) +
                      chi_squared_expectation(i)(j, li)) /
                         hyper_tau(l));
      }
    }
  }

  for (int li = 0; li < n_discrete_fields; li++) {
    int l = discrete_fields[li];

    arma::rowvec psi_li_transposed = psi(li).t();
    for (int jprime = 0; jprime < N; jprime++) {
      elbo += arma::dot(gamma(li).row(jprime),
                        psi_li_transposed - log_gamma(li).row(jprime));
    }

    elbo += beta_function(omega(l, 0), omega(l, 1), true);

    elbo += arma::dot(hyper_mu(li) - alpha(li), psi(li));
    elbo -= log_beta_constant(alpha(li), M(li));
  }

  double temp_sum_N = 0.0;
  for (int li = 0; li < n_continuous_fields; li++) {
    int l = continuous_fields[li];

    elbo += beta_function(omega(l, 0), omega(l, 1), true);

    elbo -= sigma_shape(li) * (1.0 - hyper_sigma(li, 0) / sigma_scale(li)) -
            R::lgammafn(sigma_shape(li));

    double N_rho_sum = N_double * rho_sum(l, 0);
    double sigma_inverse_mean_l = sigma_shape(li) / sigma_scale(li);

    temp_sum += log(N_rho_sum / sigma_inverse_mean_l);
    temp_sum_N -= 1.0 / N_rho_sum +
                  sigma_inverse_mean_l * pow(zeta(li, 0), 2.0) +
                  log(sigma_scale(li));

    temp_sum -= sum(sigma_inverse_mean_l *
                        (sigma_tilde.row(li) + pow(eta_tilde.row(li), 2.0) -
                         2.0 * eta_tilde.row(li)) -
                    log(sigma_tilde.row(li)));
  }

  elbo += 0.5 * (temp_sum + N_double * temp_sum_N);

  return elbo;
}
