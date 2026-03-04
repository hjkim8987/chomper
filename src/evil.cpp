#include <RcppArmadillo.h>
#include <RcppThread.h>

#include <Rcpp/Benchmark/Timer.h>

#include "evil.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppThread)]]

using namespace Rcpp;

/******************************************************************************
    Member Functions of `OptimMember` Class:
        1. Constructor and Destructor
        2. Getter and Setter
        3. Functions for CAVI optimization
******************************************************************************/

/*******************************************
    1. Constructor and Destructor
      2. Getter and Setter
      3. Functions for CAVI optimization
*******************************************/
OptimMember::OptimMember(arma::field<arma::mat> x_, int k_, arma::vec n_,
                         int N_, int p_, arma::vec discrete_fields_,
                         int n_discrete_fields_, arma::vec M_,
                         arma::vec continuous_fields_, int n_continuous_fields_,
                         arma::mat hyper_beta_,
                         arma::field<arma::vec> hyper_mu_,
                         arma::mat hyper_sigma_, arma::vec hyper_phi_,
                         arma::vec hyper_tau_, arma::vec hyper_delta_)
    : x(x_),
      k(k_),
      n(n_),
      N(N_),
      p(p_),
      discrete_fields(discrete_fields_),
      n_discrete_fields(n_discrete_fields_),
      M(M_),
      continuous_fields(continuous_fields_),
      n_continuous_fields(n_continuous_fields_),
      hyper_beta(hyper_beta_),
      hyper_mu(hyper_mu_),
      hyper_sigma(hyper_sigma_),
      hyper_phi(hyper_phi_),
      hyper_tau(hyper_tau_),
      hyper_delta(hyper_delta_) {
  N_double = (double)N;

  lambda_init = create_vector_field(k_, n_);

  alpha = create_vector_field(n_discrete_fields_, M_);
  psi = create_vector_field(n_discrete_fields_, M_);

  eta_mean = create_vector(n_continuous_fields_);
  eta_var = create_vector(n_continuous_fields_);

  sigma_shape = create_vector(n_continuous_fields_);
  sigma_scale = create_vector(n_continuous_fields_);

  omega = create_matrix(p_, 2);
  kappa = create_matrix(p_, 2);

  nu = create_matrix_field(k_, n_, N_);
  log_nu = create_matrix_field(k_, n_, N_);

  gamma = create_matrix_field(n_discrete_fields_, N_, M_);
  log_gamma = create_matrix_field(n_discrete_fields_, N_, M_);

  eta_tilde = create_matrix(n_continuous_fields_, N_);
  sigma_tilde = create_matrix(n_continuous_fields_, N_);

  chi_discrete = create_matrix_field(k_, n_, n_discrete_fields_);
  chi_expectation = create_matrix_field(k_, n_, n_continuous_fields_);
  chi_squared_expectation = create_matrix_field(k_, n_, n_continuous_fields_);

  zeta_sigma = create_matrix(n_continuous_fields_, 2);
  zeta_rho = create_matrix(n_continuous_fields_, 2);

  rho = create_matrix_field(k_, n_, p_);
  rho_sum = create_matrix(p_, 2);

  x_index = create_index_matrix(k_, n_);
  x_append = compile_matrix(x_);

  log_phi_tau = create_vector(n_discrete_fields_);
  for (int li = 0; li < n_discrete_fields_; li++) {
    int l = discrete_fields[li];

    log_phi_tau[li] = log(hyper_phi[li]) / hyper_tau[l];
  }

  log_constant_matrix = create_matrix_field(k_, n_, n_discrete_fields_);
  for (int i = 0; i < k_; i++) {
    for (int j = 0; j < n_[i]; j++) {
      for (int li = 0; li < n_discrete_fields_; li++) {
        int l = discrete_fields[li];
        double x_ijl = x[i](j, l);

        NumericVector log_level(M_[li]);
        for (int m = 1; m <= M_[li]; m++) {
          log_level(m - 1) = log_phi_tau[li] *
                             (double)(std::fabs(x_ijl - m) <= hyper_delta[li]);
        }
        log_constant_matrix[i](j, li) = log_sum_exp(log_level);
      }
    }
  }
}

OptimMember::~OptimMember() {}

/*******************************************
      1. Constructor and Destructor
    2. Getter and Setter
      3. Functions for CAVI optimization
*******************************************/
int OptimMember::getNumberOfFiles() const { return k; }
void OptimMember::setNumberOfFiles(int k_) { k = k_; }

arma::vec OptimMember::getNumberOfRecords() const { return n; }
void OptimMember::setNumberOfRecords(arma::vec n_) { n = n_; }

int OptimMember::getNumberOfTotalRecords() const { return N; }
void OptimMember::setNumberOfTotalRecords(int N_) { N = N_; }

int OptimMember::getNumberOfFields() const { return p; }
void OptimMember::setNumberOfFields(int p_) { p = p_; }

arma::vec OptimMember::getDiscreteFields() const { return discrete_fields; }
void OptimMember::setDiscreteFields(arma::vec discrete_fields_) {
  discrete_fields = discrete_fields_;
}

int OptimMember::getNumberOfDiscreteFields() const { return n_discrete_fields; }
void OptimMember::setNumberOfDiscreteFields(int n_discrete_fields_) {
  n_discrete_fields = n_discrete_fields_;
}

arma::vec OptimMember::getNumberOfLevels() const { return M; }
void OptimMember::setNumberOfLevels(arma::vec M_) { M = M_; }

arma::vec OptimMember::getContinuousFields() const { return continuous_fields; }
void OptimMember::setContinuousFields(arma::vec continuous_fields_) {
  continuous_fields = continuous_fields_;
}

int OptimMember::getNumberOfContinuousFields() const {
  return n_continuous_fields;
}
void OptimMember::setNumberOfContinuousFields(int n_continuous_fields_) {
  n_continuous_fields = n_continuous_fields_;
}

arma::field<arma::mat> OptimMember::getLogConstantMatrix() const {
  return log_constant_matrix;
}
void OptimMember::setLogConstantMatrix(
    arma::field<arma::mat> log_constant_matrix_) {
  log_constant_matrix = log_constant_matrix_;
}

double OptimMember::getLogNormalizeConstant(int i, int j, int l) const {
  return log_constant_matrix[i](j, l);
}
void OptimMember::setLogNormalizeConstant(int i, int j, int l, double value) {
  log_constant_matrix[i](j, l) = value;
}

arma::field<arma::vec> OptimMember::getApproximatedAlpha() const {
  return alpha;
}
void OptimMember::setApproximatedAlpha(arma::field<arma::vec> alpha_) {
  alpha = alpha_;
}

arma::field<arma::vec> OptimMember::getApproximatedPsi() const { return psi; }
void OptimMember::setApproximatedPsi(arma::field<arma::vec> psi_) {
  psi = psi_;
}

arma::vec OptimMember::getApproximatedEtaMean() const { return eta_mean; }
void OptimMember::setApproximatedEtaMean(arma::vec eta_mean_) {
  eta_mean = eta_mean_;
}

arma::vec OptimMember::getApproximatedEtaVar() const { return eta_var; }
void OptimMember::setApproximatedEtaVar(arma::vec eta_var_) {
  eta_var = eta_var_;
}

arma::vec OptimMember::getApproximatedSigmaShape() const { return sigma_shape; }
void OptimMember::setApproximatedSigmaShape(arma::vec sigma_shape_) {
  sigma_shape = sigma_shape_;
}

arma::vec OptimMember::getApproximatedSigmaScale() const { return sigma_scale; }
void OptimMember::setApproximatedSigmaScale(arma::vec sigma_scale_) {
  sigma_scale = sigma_scale_;
}

arma::mat OptimMember::getApproximatedOmega() const { return omega; }
void OptimMember::setApproximatedOmega(arma::mat omega_) { omega = omega_; }

arma::mat OptimMember::getApproximatedKappa() const { return kappa; }
void OptimMember::setApproximatedKappa(arma::mat kappa_) { kappa = kappa_; }

arma::field<arma::mat> OptimMember::getApproximatedNu() const { return nu; }
void OptimMember::setApproximatedNu(arma::field<arma::mat> nu_) { nu = nu_; }

arma::field<arma::mat> OptimMember::getApproximatedLogNu() const {
  return log_nu;
}
void OptimMember::setApproximatedLogNu(arma::field<arma::mat> log_nu_) {
  log_nu = log_nu_;
}

arma::field<arma::mat> OptimMember::getApproximatedGamma() const {
  return gamma;
}
void OptimMember::setApproximatedGamma(arma::field<arma::mat> gamma_) {
  gamma = gamma_;
}

arma::field<arma::mat> OptimMember::getApproximatedLogGamma() const {
  return log_gamma;
}
void OptimMember::setApproximatedLogGamma(arma::field<arma::mat> log_gamma_) {
  log_gamma = log_gamma_;
}

arma::mat OptimMember::getApproximatedEtaTilde() const { return eta_tilde; }
void OptimMember::setApproximatedEtaTilde(arma::mat eta_tilde_) {
  eta_tilde = eta_tilde_;
}

arma::mat OptimMember::getApproximatedSigmaTilde() const { return sigma_tilde; }
void OptimMember::setApproximatedSigmaTilde(arma::mat sigma_tilde_) {
  sigma_tilde = sigma_tilde_;
}

arma::field<arma::mat> OptimMember::getApproximatedRho() const { return rho; }
void OptimMember::setApproximatedRho(arma::field<arma::mat> rho_) {
  rho = rho_;
}

arma::mat OptimMember::getApproximatedRhoSum() const { return rho_sum; }
void OptimMember::setApproximatedRhoSum(arma::mat rho_sum_) {
  rho_sum = rho_sum_;
}

double OptimMember::getElapsedTime() const { return elapsed_time; }
void OptimMember::setElapsedTime(double elapsed_time_) {
  elapsed_time = elapsed_time_;
}

double OptimMember::getSamplingProb() const { return p_init; }
void OptimMember::setSamplingProb(double prob) { p_init = prob; }

double OptimMember::getELBO() const { return elbo; }
void OptimMember::setELBO(double elbo_) { elbo = elbo_; }

bool OptimMember::isInterrupted() const { return interrupted; }
void OptimMember::setInterrupted(bool interrupted_) {
  interrupted = interrupted_;
}

int OptimMember::getNumberOfIterations() const { return niter_cavi; }
void OptimMember::setNumberOfIterations(int niter_cavi_) {
  niter_cavi = niter_cavi_;
}

/*******************************************
      1. Constructor and Destructor
      2. Getter and Setter
    3. Functions for CAVI optimization
*******************************************/
void OptimMember::sum_rho_by_ij() {
  rho_sum.col(0) = sum_ij_rho_ijl(rho, k, N_double, p, false);
  rho_sum.col(1) = sum_ij_rho_ijl(rho, k, N_double, p, true);
}

void OptimMember::initialize(bool custom_initializer,
                             Nullable<List> initial_values) {
  // Set initial linkage structure:
  // Randomly assign linkage indexes to the records
  // that are identical in all non-fuzzy fields

  // 1. Find indexes of non-fuzzy fields
  IntegerVector temp_non_fuzzy_fields;
  // 1.1. Discrete Fields:
  for (int li = 0; li < n_discrete_fields; li++) {
    if (!(hyper_delta[li] > 0.0)) {
      temp_non_fuzzy_fields.push_back(discrete_fields[li]);
    }
  }

  // 1.2. Continuous_Fields:
  for (int li = 0; li < n_continuous_fields; li++) {
    int l = continuous_fields[li];
    if (!(hyper_tau[l] > 0.0001)) {
      temp_non_fuzzy_fields.push_back(l);
    }
  }

  // 1.3. Define the argument of indexes of non-fuzzy fields
  arma::vec non_fuzzy_fields = as<arma::vec>(wrap(temp_non_fuzzy_fields));

  // 1.4. Pass the argument of indexes to the initial_linkage_function
  lambda_init = initial_linkage(x_append, k, n, N, non_fuzzy_fields, x_index,
                                p_init, initial_values);

  // Initialize parameters related to y (true latent records):
  // gamma: multinomial probs for discrete fields
  gamma = init_gamma(x, lambda_init, k, n, N, M, discrete_fields,
                     n_discrete_fields, hyper_delta);
  for (int li = 0; li < n_discrete_fields; li++) {
    for (int m = 0; m < M(li); m++) {
      log_gamma(li).col(m) = log(gamma(li).col(m));
    }
  }

  // \tilde{eta}, \tilde{sigma}: mean and variance for continuous fields
  // Note that E[y_lambda_ij,l] = y_j'l and Var[y_lambda_ij,l] = 0
  // So, we initialize \tilde{eta}_j'l = y_j'l
  // and \tilde{sigma}_j'l = Var[y_.l] / N_max
  eta_tilde = init_y_mean(x, lambda_init, k, n, N, N_double, continuous_fields,
                          n_continuous_fields);
  sigma_tilde =
      init_y_var(x, k, n, N, N_double, continuous_fields, n_continuous_fields);

  // The scale parameters of sigma should be initialized
  // because it is used in the first iteration.
  // This initialization uses an approximation of
  // the original update process of sigma_scale;
  // by dividing the shape parameter by N_max,
  // instead of N_max * b^sigma_l.
  sigma_scale = init_sigma_scale(hyper_sigma, sigma_shape, eta_tilde,
                                 sigma_tilde, N_double, n_continuous_fields);

  // Initialize nu (multinomial probs for lambda)
  // based on the current linkage structure
  nu = init_nu(lambda_init, k, n, N);
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n[i]; j++) {
      log_nu(i).row(j) = log(nu(i).row(j));
    }
  }

  // Initialize rho (Bernoulli probs for z):
  // It is independent of the current linkage structure,
  // but is initialized based on the hyper-parameters
  // (Note that rho needs all the information of the current linkage structure)
  for (int l = 0; l < p; l++) {
    for (int i = 0; i < k; i++) {
      rho[i].col(l) = init_rho_ij(hyper_beta(l, 0), hyper_beta(l, 1), n[i]);
    }
  }
  sum_rho_by_ij();

  // As the shape parameter of sigma depends on E[z_ijl],
  // it should be initialized after rho is initialized
  for (int li = 0; li < n_continuous_fields; li++) {
    sigma_shape[li] = hyper_sigma(li, 0) +
                      0.5 * (N_double + rho_sum(continuous_fields[li], 0));
  }
}

void OptimMember::coordinate_ascent() {
  // Update alpha
  // q_theta ~ Dirichlet(alpha)
  alpha = update_alpha(x, hyper_mu, gamma, rho, k, discrete_fields,
                       n_discrete_fields, M);

  // Update psi_lm = digamma(alpha_lm) - digamma(sum_m alpha_lm)
  // It is an auxiliary variable used for updating gamma and calculating ELBO
  psi = update_psi(alpha, n_discrete_fields, M);

  // Update mean and variance of eta
  // q_eta ~ Normal(eta_mean, eta_var)
  eta_mean = update_eta_mean(x, rho, eta_tilde, continuous_fields,
                             n_continuous_fields, rho_sum, k, N_double);
  eta_var = update_eta_var(sigma_shape, sigma_scale, continuous_fields,
                           n_continuous_fields, rho_sum, N_double);

  // Update parameters of sigma
  // q_sigma ~ InverseGamma(sigma_shape, sigma_scale)
  sigma_shape = update_sigma_shape(hyper_sigma, rho_sum, continuous_fields,
                                   n_continuous_fields, N_double);

  // Note that zeta_sigma is needed to updated sigma_scale
  zeta_sigma =
      update_zeta(x, rho, eta_tilde, eta_var, sigma_shape, sigma_scale, rho_sum,
                  continuous_fields, n_continuous_fields, k, N_double, true);
  sigma_scale =
      update_sigma_scale(hyper_sigma, eta_tilde, sigma_tilde, x, rho,
                         zeta_sigma, continuous_fields, n_continuous_fields, k);

  // Update omega
  // q_beta ~ Beta(omega[l, 0], omega[l, 1])
  omega = update_omega(hyper_beta, rho_sum, p);

  // Update kappa_l = digamma(omega_l) - digamma(sum(omega_l))
  // It is an auxiliary variable used for updating rho
  kappa = update_kappa(omega, p);

  // Update nu
  // q_lambda ~ Multinomial(nu)
  arma::field<arma::field<arma::mat>> nu_all =
      update_nu_all(x, rho, gamma, eta_tilde, sigma_tilde, k, n, N, hyper_tau,
                    log_phi_tau, discrete_fields, n_discrete_fields, M,
                    hyper_delta, continuous_fields, n_continuous_fields);
  nu = nu_all(0);
  log_nu = nu_all(1);

  // Update gamma
  // q_y ~ Multinomial(gamma) for discrete fields
  arma::field<arma::field<arma::mat>> gamma_all =
      update_gamma(x, rho, nu, psi, k, n, N, discrete_fields, n_discrete_fields,
                   M, hyper_tau, hyper_delta, log_phi_tau);
  gamma = gamma_all(0);
  log_gamma = gamma_all(1);

  // Update chi_discrete = gamma * nu
  // It is an auxiliary variable used for updating rho
  chi_discrete = update_chi_discrete(x, gamma, nu, k, n, M, N, discrete_fields,
                                     n_discrete_fields, hyper_delta);

  // Update eta_tilde and sigma_tilde
  // q_y ~ Normal(eta_tilde, sigma_tilde) for continuous fields
  arma::field<arma::mat> y_params = update_continuous_y(
      x, rho, nu, eta_mean, sigma_shape, sigma_scale, hyper_tau, k, n, N,
      continuous_fields, n_continuous_fields);
  eta_tilde = y_params(0);
  sigma_tilde = y_params(1);

  // Update chi_continuous = nu * E[y_j'l] or nu * E[y_j'l^2]
  // They are an auxiliary variables used for updating rho
  // chi_expectation = nu * E[y_j'l]
  chi_expectation =
      update_chi_expectation(nu, eta_tilde, k, n, n_continuous_fields);
  // chi_squared_expectation = nu * E[y_j'l^2]
  chi_squared_expectation = update_chi_squared_expectation(
      nu, eta_tilde, sigma_tilde, k, n, n_continuous_fields);

  // Update rho
  // q_z ~ Bernoulli(rho)
  // Note that zeta_rho is needed to updated rho
  zeta_rho =
      update_zeta(x, rho, eta_tilde, eta_var, sigma_shape, sigma_scale, rho_sum,
                  continuous_fields, n_continuous_fields, k, N_double, false);
  rho = update_rho(x, k, n, p, discrete_fields, n_discrete_fields, M,
                   log_constant_matrix, sigma_shape, sigma_scale, psi,
                   chi_discrete, log_phi_tau, continuous_fields,
                   n_continuous_fields, chi_expectation,
                   chi_squared_expectation, zeta_rho, hyper_tau, kappa);

  // Update sum_ij rho_ij and sum_ij (1 - rho_ij) for each l
  // It is used for updating omega (parameter of beta distribution)
  rho_sum.col(0) = sum_ij_rho_ijl(rho, k, N_double, p, false);
  rho_sum.col(1) = N_double - rho_sum.col(0);

  // Save the newly calculated ELBO
  elbo = calculate_ELBO(
      x, rho, rho_sum, nu, log_nu, gamma, log_gamma, alpha, psi, omega,
      eta_tilde, sigma_tilde, sigma_shape, sigma_scale, chi_discrete,
      chi_expectation, chi_squared_expectation, zeta_sigma, hyper_mu, hyper_tau,
      hyper_sigma, log_phi_tau, log_constant_matrix, k, n, N, N_double,
      discrete_fields, n_discrete_fields, M, continuous_fields,
      n_continuous_fields);
}

void OptimMember::cavi(double eps, int max_iter, int verbose, double max_time) {
  Timer timer;
  int nano = 1e9;
  double start_t = timer.now();

  // Initialize auxiliary values for calculating ELBO
  double elbo_t0 = R_NegInf;
  double elbo_t1 = 0.0;
  double elbo_diff;

  // Main iterative optimization
  if (verbose > 1) {
    Rcpp::Rcout << "Start Coordinate Ascent Variational Inference...\n";
  }

  int niter = 0;

  double start_iter = timer.now();
  double elapsed_time_inner_loop = 0.0;
  for (int t = 0; t < max_iter; t++) {
    coordinate_ascent();

    // Extract ELBO to check convergence
    elbo_t1 = elbo;
    elbo_diff = std::fabs((elbo_t1 - elbo_t0) / elbo_t0);

    // Store the current ELBO into a temporary field
    elbo_t0 = elbo_t1;

    niter += 1;
    if (verbose > 2) {
      if (niter == 1) {
        Rcpp::Rcout << "Finished " << niter << "st iteration... (Elapsed Time: "
                    << round((timer.now() - start_iter) / nano) << " seconds)"
                    << std::endl;
      } else if ((niter % 10) == 0) {
        Rcpp::Rcout << "Finished " << niter << "th iteration... (Elapsed Time: "
                    << round((timer.now() - start_iter) / nano) << " seconds)"
                    << std::endl;
      }
    }

    // Break the loop if it converges
    if (elbo_diff < eps) {
      break;
    }

    elapsed_time_inner_loop = round((timer.now() - start_iter) / nano);
    // Break the loop if the elapsed time exceeds the maximum time
    if (elapsed_time_inner_loop > max_time) {
      Rcpp::Rcout
          << "Interrupted due to the elapsed time exceeds the maximum time"
          << std::endl;
      interrupted = true;
      break;
    }
  }

  niter_cavi = niter;
  elapsed_time = round((timer.now() - start_t) / nano * 10000.0) / 10000.0;

  if (verbose > 1) {
    Rcpp::Rcout << "- Finished CAVI:" << std::endl;
    Rcpp::Rcout << "  - Total Number of Iteration: " << niter << std::endl;
    Rcpp::Rcout << "  - Total Elapsed Time: " << elapsed_time << " seconds"
                << std::endl;
  }
}

void OptimMember::update_mutated_info() {
  // Apply mutated 'nu' to related parameters
  // The only difference between the original and mutation is 'nu',
  // so we need to updated parameters related to 'nu',
  // and update 'rho' based on the updated parameters.
  arma::field<arma::mat> rho_prev = rho;

  arma::field<arma::field<arma::mat>> gamma_all =
      update_gamma(x, rho_prev, nu, psi, k, n, N, discrete_fields,
                   n_discrete_fields, M, hyper_tau, hyper_delta, log_phi_tau);
  gamma = gamma_all(0);
  log_gamma = gamma_all(1);

  chi_discrete = update_chi_discrete(x, gamma, nu, k, n, M, N, discrete_fields,
                                     n_discrete_fields, hyper_delta);

  arma::field<arma::mat> y_params = update_continuous_y(
      x, rho_prev, nu, eta_mean, sigma_shape, sigma_scale, hyper_tau, k, n, N,
      continuous_fields, n_continuous_fields);
  eta_tilde = y_params(0);
  sigma_tilde = y_params(1);

  chi_expectation =
      update_chi_expectation(nu, eta_tilde, k, n, n_continuous_fields);
  chi_squared_expectation = update_chi_squared_expectation(
      nu, eta_tilde, sigma_tilde, k, n, n_continuous_fields);

  arma::mat zeta_mutate =
      update_zeta(x, rho, eta_tilde, eta_var, sigma_shape, sigma_scale, rho_sum,
                  continuous_fields, n_continuous_fields, k, N_double, false);
  rho = update_rho(x, k, n, p, discrete_fields, n_discrete_fields, M,
                   log_constant_matrix, sigma_shape, sigma_scale, psi,
                   chi_discrete, log_phi_tau, continuous_fields,
                   n_continuous_fields, chi_expectation,
                   chi_squared_expectation, zeta_mutate, hyper_tau, kappa);
  rho_sum.col(0) = sum_ij_rho_ijl(rho, k, N_double, p, false);
  rho_sum.col(1) = N_double - rho_sum.col(0);
}

void OptimMember::mutation_split_and_merge() {
  // Split and Merge
  // Find current linkage indexes (maximum probability)
  arma::field<arma::uvec> max_indexes = find_max_indexes(nu, k, n);

  // Randomly sample 2 records and find the linkage index of them
  arma::uvec target = sample_index(N, 2);

  arma::rowvec x1 = x_index.row(target[0]);
  arma::rowvec x2 = x_index.row(target[1]);

  unsigned int j1 = max_indexes[x1[0]][x1[1]];
  unsigned int j2 = max_indexes[x2[0]][x2[1]];

  // Find the indexes of records whose value is either j1 or j2
  std::vector<int> same_idx;
  int n_idx = 0;
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n[i]; j++) {
      if ((max_indexes[i][j] == j1) || (max_indexes[i][j] == j2)) {
        same_idx.push_back(n_idx);
      }
      n_idx += 1;
    }
  }

  if (j1 == j2) {
    // Split Procedure: j1 -> unassigned two indexes
    // Find unassigned indexes and sample two of them
    IntegerVector union_jprime;
    IntegerVector entire_index = seq(0, N - 1);
    for (int i = 0; i < k; i++) {
      IntegerVector max_int_vec = as<IntegerVector>(wrap(max_indexes[i]));
      union_jprime = union_(union_jprime, max_int_vec);
    }
    IntegerVector unassigned_indices = setdiff(entire_index, union_jprime);
    int j_new = sample(unassigned_indices, 1)(0);

    // Actual splitting the current linkage
    for (unsigned int t = 0; t < same_idx.size(); t++) {
      if (R::runif(0.0, 1.0) < 0.5) {
        arma::rowvec idx = x_index.row(same_idx[t]);
        int i = idx[0];
        int j = idx[1];

        arma::rowvec nu_ij = nu[i].row(j);
        nu[i].row(j) = swap_row_values(nu_ij, j1, j_new);
      }
    }
  } else {
    // Merge Procedure: j1 -> j2
    for (unsigned int t = 0; t < same_idx.size(); t++) {
      arma::rowvec idx = x_index.row(same_idx[t]);
      int i = idx[0];
      int j = idx[1];
      // As we are merging j1 and j2,
      // it only swaps the value of a record with j1 with j2
      if (max_indexes[i][j] == j1) {
        arma::rowvec nu_ij = nu[i].row(j);
        nu[i].row(j) = swap_row_values(nu_ij, j1, j2);
      }
    }
  }

  update_mutated_info();
}

/******************************************************************************
    Functions for Evolutionary Algorithm:
        A collection of functions handling a vector of OptimMember.
        These are not member fuctions of OptimMember class,
        because they uses multiple OptimMember's.
******************************************************************************/
std::vector<OptimMember> initialization_step(OptimMember data, int n_parents,
                                             double overlap_prob,
                                             bool custom_initializer,
                                             Nullable<List> initial_values) {
  arma::vec p_grid(n_parents);
  for (int i = 0; i < n_parents; i++) {
    p_grid[i] = R::runif(0.0, overlap_prob);
  }

  std::vector<OptimMember> P0;

  for (int u = 0; u < n_parents; u++) {
    OptimMember data_init = data;

    data_init.setSamplingProb(p_grid[u]);
    data_init.initialize(custom_initializer, initial_values);

    P0.push_back(data_init);
  }

  return P0;
}

std::vector<OptimMember> update_step(std::vector<OptimMember> P, double eps,
                                     int max_iter, int verbose, int n_threads) {
  int n_population = P.size();
  std::vector<OptimMember> P_updated;
  for (int u = 0; u < n_population; u++) {
    P_updated.push_back(P[u]);
  }

  RcppThread::parallelFor(
      0, n_population,
      [&](unsigned int i) { P_updated[i].cavi(eps, max_iter, verbose); },
      n_threads);

  return P_updated;
}

std::vector<OptimMember> crossover_step(std::vector<OptimMember> P,
                                        arma::mat combination,
                                        int n_combination, int n_children) {
  int k = P[0].getNumberOfFiles();

  arma::vec n = P[0].getNumberOfRecords();
  int N = P[0].getNumberOfTotalRecords();

  int n_discrete_fields = P[0].getNumberOfDiscreteFields();

  arma::uvec crossover_index = sample_index(n_combination, int(n_children / 2));
  arma::uvec row_index = sample_index(N, int(n_children / 2));

  std::vector<OptimMember> P2;
  for (int u = 0; u < n_children; u++) {
    P2.push_back(P[0]);
  }

  int o = 0;
  for (int u = 0; u < int(n_children / 2); u++) {
    arma::rowvec cdx = combination.row(crossover_index(u));
    int rdx = row_index(u);

    OptimMember C1 = P[cdx[0]];
    OptimMember C2 = P[cdx[1]];

    // Single-point crossover (nu)
    arma::mat nu1 = compile_matrix(C1.getApproximatedNu());
    arma::mat log_nu1 = compile_matrix(C1.getApproximatedLogNu());
    arma::mat nu2 = compile_matrix(C2.getApproximatedNu());
    arma::mat log_nu2 = compile_matrix(C2.getApproximatedLogNu());

    arma::mat nu_aux = nu1.rows(rdx, N - 1);
    nu1.rows(rdx, N - 1) = nu2.rows(rdx, N - 1);
    nu2.rows(rdx, N - 1) = nu_aux;

    arma::mat log_nu_aux = log_nu1.rows(rdx, N - 1);
    log_nu1.rows(rdx, N - 1) = log_nu2.rows(rdx, N - 1);
    log_nu2.rows(rdx, N - 1) = log_nu_aux;

    C1.setApproximatedNu(split_matrix(nu1, n, k));
    C1.setApproximatedLogNu(split_matrix(log_nu1, n, k));
    C2.setApproximatedNu(split_matrix(nu2, n, k));
    C2.setApproximatedLogNu(split_matrix(log_nu2, n, k));

    // Single-point crossover (gamma)
    arma::field<arma::mat> gamma1 = C1.getApproximatedGamma();
    arma::field<arma::mat> log_gamma1 = C1.getApproximatedLogGamma();
    arma::field<arma::mat> gamma2 = C2.getApproximatedGamma();
    arma::field<arma::mat> log_gamma2 = C2.getApproximatedLogGamma();

    for (int li = 0; li < n_discrete_fields; li++) {
      arma::mat gamma_l_aux = gamma1(li).rows(rdx, N - 1);
      gamma1(li).rows(rdx, N - 1) = gamma2(li).rows(rdx, N - 1);
      gamma2(li).rows(rdx, N - 1) = gamma_l_aux;

      arma::mat log_gamma_l_aux = log_gamma1(li).rows(rdx, N - 1);
      log_gamma1(li).rows(rdx, N - 1) = log_gamma2(li).rows(rdx, N - 1);
      log_gamma2(li).rows(rdx, N - 1) = log_gamma_l_aux;
    }

    C1.setApproximatedGamma(gamma1);
    C1.setApproximatedLogGamma(log_gamma1);
    C2.setApproximatedGamma(gamma2);
    C2.setApproximatedLogGamma(log_gamma2);

    // Single-point crossover (eta_tilde and sigma_tilde)
    arma::mat eta_tilde1 = C1.getApproximatedEtaTilde();
    arma::mat sigma_tilde1 = C1.getApproximatedSigmaTilde();
    arma::mat eta_tilde2 = C2.getApproximatedEtaTilde();
    arma::mat sigma_tilde2 = C2.getApproximatedSigmaTilde();

    arma::mat eta_tilde_aux = eta_tilde1.cols(rdx, N - 1);
    arma::mat sigma_tilde_aux = sigma_tilde1.cols(rdx, N - 1);

    eta_tilde1.cols(rdx, N - 1) = eta_tilde2.cols(rdx, N - 1);
    sigma_tilde1.cols(rdx, N - 1) = sigma_tilde2.cols(rdx, N - 1);

    eta_tilde2.cols(rdx, N - 1) = eta_tilde_aux;
    sigma_tilde2.cols(rdx, N - 1) = sigma_tilde_aux;

    C1.setApproximatedEtaTilde(eta_tilde1);
    C1.setApproximatedSigmaTilde(sigma_tilde1);

    C2.setApproximatedEtaTilde(eta_tilde2);
    C2.setApproximatedSigmaTilde(sigma_tilde2);

    // Single-point crossover (rho)
    arma::mat rho1 = compile_matrix(C1.getApproximatedRho());
    arma::mat rho2 = compile_matrix(C2.getApproximatedRho());

    arma::mat rho_aux = rho1.rows(rdx, N - 1);
    rho1.rows(rdx, N - 1) = rho2.rows(rdx, N - 1);
    rho2.rows(rdx, N - 1) = rho_aux;

    C1.setApproximatedRho(split_matrix(rho1, n, k));
    C1.sum_rho_by_ij();

    C2.setApproximatedRho(split_matrix(rho2, n, k));
    C2.sum_rho_by_ij();

    P2[o] = C1;
    P2[o + 1] = C2;

    o += 2;
  }

  return P2;
}

std::vector<OptimMember> selection_step(std::vector<OptimMember>& P1,
                                        std::vector<OptimMember>& P3,
                                        int n_parents, int n_children) {
  // Get ELBOs from both parents and children,
  // and combine them into a single vector
  NumericVector elbo_parents(n_parents);
  for (int i = 0; i < n_parents; i++) {
    elbo_parents[i] = P1[i].getELBO();
  }

  NumericVector elbo_children(n_children);
  for (int i = 0; i < n_children; i++) {
    elbo_children[i] = P3[i].getELBO();
  }

  // Find the top "n_parents" indexes from the combined ELBO vector
  IntegerVector sorted_index =
      get_top_n_indexes(elbo_parents, elbo_children, n_parents);

  // Extract the indices of the top 'n_parents' parents and children
  std::vector<OptimMember> P4;
  for (int i = 0; i < n_parents; i++) {
    int idx = sorted_index(i);

    if (idx < n_parents) {
      P4.push_back(P1[idx]);
    } else {
      P4.push_back(P3[idx - n_parents]);
    }
  }

  return P4;
}

std::vector<OptimMember> mutation_step(std::vector<OptimMember> P4,
                                       int n_parents, int n_threads) {
  std::vector<OptimMember> P5;
  for (int u = 0; u < n_parents; u++) {
    P5.push_back(P4[u]);
  }

  for (int i = 0; i < n_parents; i++) {
    P5[i].mutation_split_and_merge();
  }

  return P5;
}