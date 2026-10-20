#include <RcppArmadillo.h>

#include "gibbs.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

//// Generating MCMC samples from the full conditional distributions
// beta_{l} ~ Beta;
double rbeta_l(double a, double b, arma::field<arma::mat> z, int l, int k,
               int N) {
  double z_sum = 0.0;
  for (int i = 0; i < k; i++) {
    z_sum += sum(z(i).col(l));
  }
  return R::rbeta(a + z_sum, b + (N - z_sum));
}

// theta_{l} ~ Dirichlet;
NumericVector rtheta_l(NumericVector mu, arma::field<arma::mat> x, arma::mat y,
                       arma::field<arma::mat> z, int l, int k, int M_l) {
  NumericVector alpha = mu;

  for (int m = 0; m < M_l; m++) {
    alpha(m) += sum(y.col(l) == (m + 1));
    for (int i = 0; i < k; i++) {
      alpha(m) += sum(z(i).col(l) % (x(i).col(l) == (m + 1)));
    }
  }

  return rdirichlet(alpha);
}

// y_{jprime, l} ~ Multinomial; for l = 1, ..., l_1
int ry_jprimel_multinomial(double phi_l, double tau_l, arma::field<arma::mat> x,
                           arma::field<IntegerVector> lambda,
                           arma::field<arma::mat> z, NumericVector theta_l,
                           int jprime, int l, int k, arma::vec n, int M_l,
                           arma::field<arma::field<arma::mat>> x_in_range,
                           int l_original) {
  NumericVector u(M_l);
  for (int m = 0; m < M_l; m++) {
    double expo = 0.0;
    for (int i = 0; i < k; i++) {
      arma::colvec lambda_i_is_jprime =
          as<arma::vec>(wrap(lambda(i) == jprime));
      expo += sum(x_in_range(i)(l_original).col(m) % lambda_i_is_jprime %
                  (1.0 - z(i).col(l)));
    }
    u(m) = expo / tau_l * log(phi_l) + log(theta_l(m));
  }

  NumericVector probs = log_softmax(u, false);

  return rcategorical(probs, M_l);
}

// y_{jprime, l} ~ Normal; for l = l_1 + 1, ..., p
double ry_jprimel_normal(double tau_l, int jprime, int l, int k, arma::vec n,
                         arma::field<arma::mat> x, arma::field<arma::mat> z,
                         arma::field<IntegerVector> lambda, double eta_l,
                         double sigma_l) {
  double a = 0.0;
  double b = 0.0;
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n(i); j++) {
      if (lambda(i)(j) == jprime) {
        double weight = 1.0 - z(i)(j, l);
        a += weight * x(i)(j, l);
        b += weight;
      }
    }
  }

  double numerator = sigma_l * a + eta_l * tau_l;
  double denominator = sigma_l * b + tau_l;

  return R::rnorm(numerator / denominator,
                  std::sqrt(tau_l * sigma_l / denominator));
}

// eta_{l} ~ Normal;
double reta_l(arma::field<arma::mat> x, arma::field<arma::mat> z,
              arma::colvec y_l, double sigma_l, int N, int l, int k) {
  double zx_sum = 0.0;
  double z_sum = 0.0;
  for (int i = 0; i < k; i++) {
    zx_sum += sum(x(i).col(l) % z(i).col(l));
    z_sum += sum(z(i).col(l));
  }

  double denominator = (double)(N) + z_sum;

  return R::rnorm((sum(y_l) + zx_sum) / denominator,
                  std::sqrt(sigma_l / denominator));
}

// sigma_{l} ~ Inverse-Gamma;
double rsigma_l(arma::rowvec hyper_sigma_l, arma::colvec y_l, double eta_l,
                arma::field<arma::mat> x, arma::field<arma::mat> z, int l,
                int k, int N) {
  double zx_sum = 0.0;
  double z_sum = 0.0;
  for (int i = 0; i < k; i++) {
    zx_sum += sum(z(i).col(l) % pow(x(i).col(l) - eta_l, 2.0));
    z_sum += sum(z(i).col(l));
  }

  return rinvgamma(
      hyper_sigma_l(0) + 0.5 * (N + z_sum),
      hyper_sigma_l(1) + 0.5 * (sum(pow(y_l - eta_l, 2)) + zx_sum));
}

// z_{ijl} ~ Bernoulli; for l = 1, ..., l_1
int rz_ijl_multinomial(double beta_l, double theta_l, double x_probability_yx) {
  double a = beta_l * theta_l;
  double b = (1.0 - beta_l) * x_probability_yx;
  return rbernoulli(a / (a + b));
}

// z_{ijl} ~ Bernoulli; for l = l_1 + 1, ..., p
int rz_ijl_normal(double tau_l, double x_ijl, double beta_l, double y_jprimel,
                  double eta_l, double sigma_l) {
  double a = beta_l * R::dnorm(x_ijl, eta_l, sqrt(sigma_l), false);
  double b = (1.0 - beta_l) * R::dnorm(x_ijl, y_jprimel, sqrt(tau_l), false);

  return rbernoulli(a / (a + b));
}

//// Code for split and merge algorithm
// Get log-likelihood with the given MCMC samples,
// which is used to check the acceptance of split (or merge) process
double get_loglikelihood(arma::rowvec z, int lambda_ij, int j1,
                         arma::rowvec y_j1, int j2, arma::rowvec y_j2,

                         arma::rowvec x,
                         arma::field<arma::mat> log_x_probability,

                         arma::field<NumericVector> log_theta,
                         NumericVector log_odds, arma::field<NumericVector> mu,

                         NumericVector eta, NumericVector sigma,
                         arma::vec hyper_tau, arma::vec discrete_fields,
                         int n_discrete_fields, arma::vec M,
                         arma::vec continuous_fields, int n_continuous_fields) {
  double ell = 0.0;

  // Log-likelihood for discrete fields
  for (int l = 0; l < n_discrete_fields; l++) {
    int ldx = discrete_fields(l);

    for (int m = 0; m < M(l); m++) {
      ell += (double)(x(ldx) == (m + 1)) * z(ldx) * log_theta(l)(m);
    }

    // The log-likelihood is calculated with corresponding assignment of x_{ij}.
    if (lambda_ij == j1) {
      // If x_{ij} is assigned to j1, i.e., lambda_{ij} = j1,
      // the log-likelihood is calculated with y_{j1}
      ell += (1.0 - z(ldx)) * log_x_probability(l)(y_j1(ldx) - 1, x(ldx) - 1);
    } else if (lambda_ij == j2) {
      // If x_{ij} is assigned to j2, i.e., lambda_{ij} = j2,
      // the log-likelihood is calculated with y_{j2}
      ell += (1.0 - z(ldx)) * log_x_probability(l)(y_j2(ldx) - 1, x(ldx) - 1);
    }

    ell += z(ldx) * log_odds(ldx);
  }

  // Log-likelihood for continuous fields
  for (int l = 0; l < n_continuous_fields; l++) {
    int ldx = continuous_fields(l);

    // The log-likelihood is calculated with corresponding assignment of x_{ij}.
    if (lambda_ij == j1) {
      // If x_{ij} is assigned to j1, i.e., lambda_{ij} = j1,
      // the log-likelihood is calculated with y_{j1}
      ell += (1.0 - z(ldx)) *
             R::dnorm(x(ldx), y_j1(ldx), sqrt(hyper_tau(ldx)), true);
    } else if (lambda_ij == j2) {
      // If x_{ij} is assigned to j2, i.e., lambda_{ij} = j2,
      // the log-likelihood is calculated with y_{j2}
      ell += (1.0 - z(ldx)) *
             R::dnorm(x(ldx), y_j2(ldx), sqrt(hyper_tau(ldx)), true);
    }

    ell += z(ldx) * R::dnorm(x(ldx), eta(l), sqrt(sigma(l)), true);

    ell += z(ldx) * log_odds(ldx);
  }

  return ell;
}

// Get log-likelihood for latent records with the given MCMC samples,
// which is used to check the acceptance of split (or merge) process
double get_loglikelihood_latent(
    arma::rowvec y, arma::field<NumericVector> log_theta, NumericVector eta,
    NumericVector sigma, arma::vec discrete_fields, int n_discrete_fields,
    arma::vec continuous_fields, int n_continuous_fields, arma::vec M) {
  double ell = 0.0;

  for (int l = 0; l < n_discrete_fields; l++) {
    int ldx = discrete_fields(l);

    for (int m = 0; m < M(l); m++) {
      ell += (double)(y(ldx) == (m + 1)) * log_theta(l)(m);
    }
  }

  for (int l = 0; l < n_continuous_fields; l++) {
    int ldx = continuous_fields(l);

    ell += R::dnorm(y(ldx), eta(l), sqrt(sigma(l)), true);
  }

  return ell;
}

// Create a matrix C of information for splitting records
// With the indexes of records, (i, j), linked to j',
// and the indexes of the selected pair of records, (i1, j1) and (i2, j2),
// randomly reassign (split) the records to either j1 or j2
arma::mat create_split_collection(IntegerVector file_index,
                                  IntegerVector location_index,
                                  IntegerVector file_selected,
                                  IntegerVector location_selected, int j1,
                                  int j2) {
  // Define a matrix C to contain the information of splitting records
  // Fill 'nrow - 2' rows first, and append two rows for the selected pair
  int nrow_m2 = file_index.size();
  arma::mat C(nrow_m2 + 2, 3);
  for (int i = 0; i < nrow_m2; i++) {
    C(i, 0) = file_index(i);
    C(i, 1) = location_index(i);
    // Randomly assign the record to either j1 or j2
    if (R::runif(0.0, 1.0) < 0.5) {
      C(i, 2) = j1;
    } else {
      C(i, 2) = j2;
    }
  }

  // Append the selected pair
  C(nrow_m2, 0) = file_selected(0);
  C(nrow_m2, 1) = location_selected(0);
  C(nrow_m2 + 1, 0) = file_selected(1);
  C(nrow_m2 + 1, 1) = location_selected(1);

  // Randomly assign the selected pair to either j1 or j2
  if (R::runif(0.0, 1.0) < 0.5) {
    C(nrow_m2, 2) = j1;
    C(nrow_m2 + 1, 2) = j2;
  } else {
    C(nrow_m2, 2) = j2;
    C(nrow_m2 + 1, 2) = j1;
  }

  return C;
}

// Update distortion indicator z_{ij}
// with the current MCMC samples and the new assignments
arma::rowvec update_distortion(arma::rowvec x_ij, arma::rowvec y_jprime,
                               NumericVector beta,
                               arma::field<NumericVector> theta,
                               NumericVector eta, NumericVector sigma, int p,
                               arma::vec discrete_fields, int n_discrete_fields,
                               arma::vec continuous_fields,
                               int n_continuous_fields, arma::vec hyper_tau,
                               arma::field<arma::mat> x_probability) {
  arma::rowvec z_ij(p, arma::fill::zeros);
  for (int l = 0; l < n_discrete_fields; l++) {
    // Update z_{ijl} for l = 1, ..., l_1
    int ldx = discrete_fields(l);
    z_ij(ldx) =
        rz_ijl_multinomial(beta(ldx), theta(l)(x_ij(ldx) - 1),
                           x_probability(l)(y_jprime(ldx) - 1, x_ij(ldx) - 1));
  }

  for (int l = 0; l < n_continuous_fields; l++) {
    // Update z_{ijl} for l = l_1 + 1, ..., p
    int ldx = continuous_fields(l);
    z_ij(ldx) = rz_ijl_normal(hyper_tau(ldx), x_ij(ldx), beta(ldx),
                              y_jprime(ldx), eta(l), sigma(l));
  }

  return z_ij;
}

// Update true latent record y_{jprime}
// with the current MCMC samples, new assignments,
// and the candidates of z_{ijl}'s
arma::rowvec update_latent_record(
    arma::vec hyper_phi, arma::vec hyper_tau, arma::field<arma::mat> x,
    arma::field<IntegerVector> lambda, arma::field<arma::mat> z,
    arma::field<NumericVector> theta, NumericVector eta, NumericVector sigma,
    int jprime, int k, arma::vec n, arma::vec M, int p,
    arma::vec discrete_fields, int n_discrete_fields,
    arma::vec continuous_fields, int n_continuous_fields,
    arma::field<arma::field<arma::mat>> x_in_range) {
  arma::rowvec y_jprime(p, arma::fill::zeros);

  for (int l = 0; l < n_discrete_fields; l++) {
    int ldx = discrete_fields(l);
    // Update y_{jprime, l} for l = 1, ..., l_1
    y_jprime(ldx) = ry_jprimel_multinomial(hyper_phi(l), hyper_tau(ldx), x,
                                           lambda, z, theta(l), jprime, ldx, k,
                                           n, M(l), x_in_range, l);
  }

  for (int l = 0; l < n_continuous_fields; l++) {
    // Update y_{jprime, l} for l = l_1 + 1, ..., p
    int ldx = continuous_fields(l);
    y_jprime(ldx) = ry_jprimel_normal(hyper_tau(ldx), jprime, ldx, k, n, x, z,
                                      lambda, eta(l), sigma(l));
  }

  return y_jprime;
}

// Split linked records
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
    arma::field<arma::field<arma::mat>> x_in_range, bool sample_y) {
  // Define a field to contain the results of split process
  arma::field<arma::mat> split_result(5);

  int jprime = lambda_ij;
  IntegerVector union_jprime;

  // Convert record index to string, for example, "i.j"
  // This is just for easier handling of record indexes
  String r1 =
      convert_pair_to_string(selected_files(0), selected_records_index(0));
  String r2 =
      convert_pair_to_string(selected_files(1), selected_records_index(1));

  // Find all other records assigned to j' except the selected pair,
  // because we should split "All records assigned to j'"
  // into two different latent records
  // C_file: file indexes of records assigned to j'
  // C_location: record indexes of records assigned to j'
  IntegerVector C_file;
  IntegerVector C_location;
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n(i); j++) {
      // If a record is linked to y_{jprime, l},
      if (lambda(i)(j) == jprime) {
        String c_ij = convert_pair_to_string(i, j);
        // and if the record is not one of the selected pair,
        // append the index of the record to C_file and C_location
        if ((r1 != c_ij) && (r2 != c_ij)) {
          C_file.push_back(i);
          C_location.push_back(j);
        }
      }
    }
    // Union currently assigned j':
    // It will be used to sample an unassigned record
    // that records will be split into.
    union_jprime = union_(union_jprime, lambda(i));
  }

  // Find all unassigned records
  IntegerVector unassigned_indices = setdiff(entire_index, union_jprime);

  // Sample an unassigned record into which records will be split
  // j1: the index of the first split group
  // j2: the index of the second split group
  int j1 = jprime;
  int j2 = sample(unassigned_indices, 1)(0);

  // Get a matrix C of information for splitting records
  arma::mat C = create_split_collection(C_file, C_location, selected_files,
                                        selected_records_index, j1, j2);

  // Get row numbers of C for records assigned to j1 and j2
  IntegerVector j1_loc;
  IntegerVector j2_loc;
  int n_rows_of_C = C.n_rows;
  for (int c = 0; c < n_rows_of_C; c++) {
    if (C(c, 2) == j1) {
      j1_loc.push_back(c);
    } else {
      j2_loc.push_back(c);
    }
  }

  // Among row numbers of C for records assigned to j1 and j2,
  // sample one record x_{ijl} and replace y_{j1} with it,
  // and sample one record x_{ijl} and replace y_{j2} with it
  int new_y_idx1 = sample(j1_loc, 1)(0);
  int new_y_idx2 = sample(j2_loc, 1)(0);

  arma::rowvec y_j1 = x(C(new_y_idx1, 0)).row(C(new_y_idx1, 1));
  arma::rowvec y_j2 = x(C(new_y_idx2, 0)).row(C(new_y_idx2, 1));

  // Based on the new assignments, sample corresponding z_{ijl}'s
  arma::mat z_candidate(n_rows_of_C, p, arma::fill::zeros);
  for (int c = 0; c < n_rows_of_C; c++) {
    if (C(c, 2) == j1) {
      z_candidate.row(c) = update_distortion(
          x(C(c, 0)).row(C(c, 1)), y_j1, beta, theta, eta, sigma, p,
          discrete_fields, n_discrete_fields, continuous_fields,
          n_continuous_fields, hyper_tau, x_probability);
    } else {
      z_candidate.row(c) = update_distortion(
          x(C(c, 0)).row(C(c, 1)), y_j2, beta, theta, eta, sigma, p,
          discrete_fields, n_discrete_fields, continuous_fields,
          n_continuous_fields, hyper_tau, x_probability);
    }
  }

  if (sample_y) {
    // New true latent record y_{jprime} should be generated
    // with the new z_{ijl}'s and lambda_{ij}'s
    // This is only for generating candidates of y_{jprime}
    // Acceptance will be decided later based on the log-likelihood
    arma::field<arma::mat> z_candidate_all = z;
    arma::field<IntegerVector> lambda_candidate_all(k);
    for (int i = 0; i < k; i++) {
      lambda_candidate_all(i) = clone(lambda(i));
    }

    for (int c = 0; c < n_rows_of_C; c++) {
      // Update new candidate lambda' into current lambda
      lambda_candidate_all(C(c, 0))(C(c, 1)) = C(c, 2);
      // Update new candidate z' into current z
      z_candidate_all(C(c, 0)).row(C(c, 1)) = z_candidate.row(c);
    }

    // Sample new y_{j1} and y_{j2}
    // with the candidates of z_{ijl}'s and lambda_{ij}'s
    y_j1 = update_latent_record(
        hyper_phi, hyper_tau, x, lambda_candidate_all, z_candidate_all, theta,
        eta, sigma, j1, k, n, M, p, discrete_fields, n_discrete_fields,
        continuous_fields, n_continuous_fields, x_in_range);
    y_j2 = update_latent_record(
        hyper_phi, hyper_tau, x, lambda_candidate_all, z_candidate_all, theta,
        eta, sigma, j2, k, n, M, p, discrete_fields, n_discrete_fields,
        continuous_fields, n_continuous_fields, x_in_range);
  }
  // Calculate log-likelihood of both current and candidate
  double ell_old = 0.0;
  ell_old += get_loglikelihood_latent(
      y.row(j1), log_theta, eta, sigma, discrete_fields, n_discrete_fields,
      continuous_fields, n_continuous_fields, M);
  ell_old += get_loglikelihood_latent(
      y.row(j2), log_theta, eta, sigma, discrete_fields, n_discrete_fields,
      continuous_fields, n_continuous_fields, M);

  double ell_new = 0.0;
  ell_new += get_loglikelihood_latent(
      y_j1, log_theta, eta, sigma, discrete_fields, n_discrete_fields,
      continuous_fields, n_continuous_fields, M);
  ell_new += get_loglikelihood_latent(
      y_j2, log_theta, eta, sigma, discrete_fields, n_discrete_fields,
      continuous_fields, n_continuous_fields, M);

  for (int c = 0; c < n_rows_of_C; c++) {
    // All y, z, and lambda remain the same
    // except for the values corresponding to (i, j)'s in a collection, C.
    // That is, log-likelihood only for the values in C should be calculated,
    // because the rest will be cancelled out when calculating the difference.
    //
    // Note that:
    // z(C(c, 0)).row(C(c, 1)): z_{ij} of a current distortion of (i, j)
    // lambda(C(c, 0))(C(c, 1)): j' = j1, the current assignment of lambda_{ij}
    // y.row(j1): y_{j1} of a current true latent record
    // y.row(j2): y_{j2} of a current true latent record
    ell_old += get_loglikelihood(
        z(C(c, 0)).row(C(c, 1)), lambda(C(c, 0))(C(c, 1)), j1, y.row(j1), j2,
        y.row(j2), x(C(c, 0)).row(C(c, 1)), log_x_probability, log_theta,
        log_odds, mu, eta, sigma, hyper_tau, discrete_fields, n_discrete_fields,
        M, continuous_fields, n_continuous_fields);
    // z_candidate.row(c): z'_{ij} of a candidate distortion of (i, j)
    // C(c, 2): either j1 or j2, the candidate assignment of lambda'_{ij}
    // y_j1: y'_{j1} of a candidate true latent record,
    //       if x_{ij} is newly assigned to j1
    // y_j2: y'_{j2} of a candidate true latent record,
    //       if x_{ij} is newly assigned to j2
    ell_new += get_loglikelihood(
        z_candidate.row(c), C(c, 2), j1, y_j1, j2, y_j2,
        x(C(c, 0)).row(C(c, 1)), log_x_probability, log_theta, log_odds, mu,
        eta, sigma, hyper_tau, discrete_fields, n_discrete_fields, M,
        continuous_fields, n_continuous_fields);
  }

  // Adjusted acceptance probability due to split and merge
  arma::mat acceptance(1, 1);
  if (log(R::runif(0.0, 1.0)) <=
      ((n_rows_of_C - 2.0) * log(2.0) + ell_new - ell_old)) {
    // Accept
    acceptance(0, 0) = 1.0;
  } else {
    // Reject
    acceptance(0, 0) = 0.0;
  }
  split_result(0) = acceptance;

  split_result(1) = C;
  arma::mat y_candidate(2, p);
  for (int l = 0; l < p; l++) {
    y_candidate(0, l) = y_j1(l);
    y_candidate(1, l) = y_j2(l);
  }
  split_result(2) = y_candidate;
  split_result(3) = z_candidate;
  arma::mat y_loc(2, 1);
  y_loc(0, 0) = j1;
  y_loc(1, 0) = j2;
  split_result(4) = y_loc;

  return split_result;
}

// Merge records
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
    arma::field<arma::field<arma::mat>> x_in_range, bool sample_y) {
  // Define a field to contain the results of merge process
  arma::field<arma::mat> merge_result(5);

  // Convert record index to string, for example, "i.j"
  // This is just for easier handling of record indexes
  String r1 =
      convert_pair_to_string(selected_files(0), selected_records_index(0));
  String r2 =
      convert_pair_to_string(selected_files(1), selected_records_index(1));

  // Find all other records assigned to either lambda1 or lambda2
  // except the selected pair,
  // because we should merge "All records assigned to lambda1 or lambda2"
  // into a single latent record
  // C_file: file indexes of records assigned to lambda1 or lambda2
  // C_location: record indexes of records assigned to lambda1 or lambda2
  IntegerVector C_file;
  IntegerVector C_location;
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n(i); j++) {
      // If a record is assigned to either lambda1 or lambda2,
      if ((lambda(i)(j) == lambda1) || (lambda(i)(j) == lambda2)) {
        // Append the index of the record to C_file and C_location
        C_file.push_back(i);
        C_location.push_back(j);
      }
    }
  }

  // Merge all the records into y_{lambda2}
  // We don't need to resample from (lambda1, lambda2),
  // because both are already chosen randomly
  int jprime = lambda2;

  // Create a matrix C of information of merging records
  int C_size = C_file.size();
  IntegerVector C_jprime = rep(jprime, C_size);

  arma::mat C(C_size, 3);
  C.col(0) = as<arma::colvec>(wrap(C_file));
  C.col(1) = as<arma::colvec>(wrap(C_location));
  C.col(2) = as<arma::colvec>(wrap(C_jprime));

  // Assign a merged latent record by sampling x_{ijl}
  arma::rowvec y_jprime = x(selected_files(1)).row(selected_records_index(1));

  // Based on the new assignments, sample corresponding z_{ijl}'s
  int n_rows_of_C = C.n_rows;
  arma::mat z_candidate(n_rows_of_C, p, arma::fill::zeros);
  for (int c = 0; c < n_rows_of_C; c++) {
    z_candidate.row(c) = update_distortion(
        x(C(c, 0)).row(C(c, 1)), y_jprime, beta, theta, eta, sigma, p,
        discrete_fields, n_discrete_fields, continuous_fields,
        n_continuous_fields, hyper_tau, x_probability);
  }

  if (sample_y) {
    // New true latent record y_{jprime} should be generated
    // with the new z_{ijl}'s and lambda_{ij}
    // This is only for generating candidates of y_{jprime}
    // Acceptance will be decided later based on the log-likelihood
    arma::field<arma::mat> z_candidate_all = z;
    arma::field<IntegerVector> lambda_candidate_all(k);
    for (int i = 0; i < k; i++) {
      lambda_candidate_all(i) = clone(lambda(i));
    }

    for (int c = 0; c < n_rows_of_C; c++) {
      // Update new candidate lambda' into current lambda
      lambda_candidate_all(C(c, 0))(C(c, 1)) = C(c, 2);
      // Update new candidate z' into current z
      z_candidate_all(C(c, 0)).row(C(c, 1)) = z_candidate.row(c);
    }

    // Sample new y_{jprime}
    // with the candidates of z_{ijl}'s and lambda_{ij}
    y_jprime = update_latent_record(
        hyper_phi, hyper_tau, x, lambda_candidate_all, z_candidate_all, theta,
        eta, sigma, jprime, k, n, M, p, discrete_fields, n_discrete_fields,
        continuous_fields, n_continuous_fields, x_in_range);
  }

  // Calculate log-likelihood of both current and candidate
  double ell_old = 0.0;
  ell_old += get_loglikelihood_latent(
      y.row(lambda2), log_theta, eta, sigma, discrete_fields, n_discrete_fields,
      continuous_fields, n_continuous_fields, M);

  double ell_new = 0.0;
  ell_new += get_loglikelihood_latent(
      y_jprime, log_theta, eta, sigma, discrete_fields, n_discrete_fields,
      continuous_fields, n_continuous_fields, M);

  for (int c = 0; c < n_rows_of_C; c++) {
    // All y, z, and lambda remain the same
    // except for the values corresponding to (i, j)'s in a collection, C.
    // That is, log-likelihood only for the values in C should be calculated,
    // because the rest will be cancelled out when calculating the difference.
    //
    // Note that:
    // z(C(c, 0)).row(C(c, 1)): z_{ij} of a current distortion of (i, j)
    // lambda(C(c, 0))(C(c, 1)): either lambda1 or lambda2,
    //                           the current assignment of lambda_{ij}
    //                           it can be one of them, because we are merging
    //                           two sampled records into the same latent record
    // y.row(lambda1): y_{lambda1} of a current true latent record
    // y.row(lambda2): y_{lambda2} of a current true latent record
    ell_old += get_loglikelihood(
        z(C(c, 0)).row(C(c, 1)), lambda(C(c, 0))(C(c, 1)), lambda1,
        y.row(lambda1), lambda2, y.row(lambda2), x(C(c, 0)).row(C(c, 1)),
        log_x_probability, log_theta, log_odds, mu, eta, sigma, hyper_tau,
        discrete_fields, n_discrete_fields, M, continuous_fields,
        n_continuous_fields);
    // z_candidate.row(c): z'_{ij} of a candidate distortion of (i, j)
    // jprime: lambda2, the candidate assignment of lambda'_{ij}
    // y_jprime: y'_{jprime} of a candidate true latent record,
    //           note that x_{ij}'s are newly assigned to y_{jprime},
    //           because all lambda'_{ij} = jprime = lambda2
    ell_new += get_loglikelihood(
        z_candidate.row(c), jprime, jprime, y_jprime, jprime, y_jprime,
        x(C(c, 0)).row(C(c, 1)), log_x_probability, log_theta, log_odds, mu,
        eta, sigma, hyper_tau, discrete_fields, n_discrete_fields, M,
        continuous_fields, n_continuous_fields);
  }

  // Adjusted acceptance probability due to split and merge
  arma::mat acceptance(1, 1);
  if (log(R::runif(0.0, 1.0)) <=
      ((2.0 - n_rows_of_C) * log(2.0) + ell_new - ell_old)) {
    // Accept
    acceptance(0, 0) = 1.0;
  } else {
    // Reject
    acceptance(0, 0) = 0.0;
  }
  merge_result(0) = acceptance;

  merge_result(1) = C;
  // Convert 'arma::rowvec' to a 1 by p matrix
  arma::mat y_candidate(1, p);
  for (int l = 0; l < p; l++) {
    y_candidate(0, l) = y_jprime(l);
  }
  merge_result(2) = y_candidate;
  merge_result(3) = z_candidate;
  arma::mat y_loc(1, 1);
  y_loc(0, 0) = jprime;
  merge_result(4) = y_loc;

  return merge_result;
}

// Do split and merge process
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
    arma::field<arma::field<arma::mat>> x_in_range, bool sample_y) {
  // Split and Merge Metropolis-Hastings result
  arma::field<arma::mat> split_merge(5);

  // Sample pairs of records from different files
  // selected_files: a vector of two files
  // selected_records_index: a vector of two records (indexes of each file)
  // lambda_i1j1: latent individual index of the first record
  // lambda_i2j2: latent individual index of the second record
  IntegerVector selected_files = sample(files, 2);
  IntegerVector selected_records_index(2);
  for (int i = 0; i < 2; i++) {
    selected_records_index(i) = sample(indices(selected_files(i)), 1)(0);
  }
  int lambda_i1j1 = lambda(selected_files(0))(selected_records_index(0));
  int lambda_i2j2 = lambda(selected_files(1))(selected_records_index(1));

  // Choose which process to do; either split or merge
  if (lambda_i1j1 == lambda_i2j2) {
    // Two records are currently assigned to the same latent individual
    // -> Split linked records into two different latent records
    split_merge =
        do_split(lambda_i1j1, selected_files, selected_records_index,
                 entire_index, k, n, p, M, x, y, z, lambda, beta, theta, eta,
                 sigma, log_odds, log_theta, discrete_fields, n_discrete_fields,
                 continuous_fields, n_continuous_fields, hyper_phi, hyper_tau,
                 mu, x_probability, log_x_probability, x_in_range, sample_y);
  } else {
    // Two records are currently assigned to different latent individuals
    // -> Merge records into the same latent record
    split_merge = do_merge(
        lambda_i1j1, lambda_i2j2, selected_files, selected_records_index, k, n,
        p, M, x, y, z, lambda, beta, theta, eta, sigma, log_odds, log_theta,
        discrete_fields, n_discrete_fields, continuous_fields,
        n_continuous_fields, hyper_phi, hyper_tau, mu, x_probability,
        log_x_probability, x_in_range, sample_y);
  }

  return split_merge;
}