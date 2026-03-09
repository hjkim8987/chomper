#include <RcppArmadillo.h>
#include <RcppThread.h>

#include <Rcpp/Benchmark/Timer.h>

#include "inference.h"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppThread)]]

using namespace Rcpp;

// [[Rcpp::export(name = ".EvolutionaryVI")]]
List EvolutionaryVI(arma::field<arma::mat> x, int k, arma::vec n, int N, int p,
                    arma::vec discrete_fields, int n_discrete_fields,
                    arma::vec M, arma::vec continuous_fields,
                    int n_continuous_fields, arma::mat hyper_beta,
                    arma::mat hyper_sigma, arma::vec hyper_phi,
                    arma::vec hyper_tau, arma::vec hyper_delta,
                    double overlap_prob, int n_parents, int n_children,
                    double tol_cavi, int max_iter_cavi, double tol_evi,
                    int max_iter_evi, bool verbose, int n_threads,
                    double max_time, bool custom_initializer,
                    bool use_checkpoint, Nullable<List> initial_values,
                    Nullable<List> checkpoint_values) {
  Timer timer;
  int nano = 1e9;
  double start_t = timer.now();

  arma::mat combination_matrix = create_combination(n_parents);
  int n_combination = combination_matrix.n_rows;

  // theta_{l} ~ Dirichlet(mu_{l}), where mu_{l} is rep(1, M(l))
  arma::field<arma::vec> hyper_mu(n_discrete_fields);
  for (int li = 0; li < n_discrete_fields; li++) {
    arma::vec mu_each(M(li), arma::fill::ones);

    hyper_mu(li) = mu_each;
  }

  // Create Initial Data Class
  OptimMember data(x, k, n, N, p, discrete_fields, n_discrete_fields, M,
                   continuous_fields, n_continuous_fields, hyper_beta, hyper_mu,
                   hyper_sigma, hyper_phi, hyper_tau, hyper_delta);

  // Main iterative optimization
  double start_iter = timer.now();
  if (verbose) {
    Rcpp::Rcout << "Start Evolutionary Variational Inference..." << std::endl;
  }

  NumericVector ELBO_best;
  int niter = 0;
  double maximum_elapsed_time = 0.0;

  // 1. Initialize: P_0
  if (custom_initializer) {
    // lambda_{ij} are prespecified based on the domain knowledge.
    if (initial_values.isNull()) {
      if (verbose) {
        Rcpp::Rcout << "---------------------------------------" << std::endl;
        Rcpp::Rcout << "- ERROR: No initial values provided;   " << std::endl;
        Rcpp::Rcout << "         Discard the current results. -" << std::endl;
        Rcpp::Rcout << "---------------------------------------" << std::endl;
      }

      return (List::create(Named("error") = "No initial values provided"));
    }
  }
  std::vector<OptimMember> P0 = initialization_step(
      data, n_parents, overlap_prob, custom_initializer, initial_values);

  if (use_checkpoint) {
    if (verbose) {
      Rcpp::Rcout << " - Loading checkpoint..." << std::endl;
      Rcpp::Rcout
          << " - Overriding previous approximations as initial values..."
          << std::endl;
    }

    if (checkpoint_values.isNull()) {
      if (verbose) {
        Rcpp::Rcout << "----------------------------------------" << std::endl;
        Rcpp::Rcout << "- ERROR: No checkpoint values provided; " << std::endl;
        Rcpp::Rcout << "         Discard the current results.  -" << std::endl;
        Rcpp::Rcout << "----------------------------------------" << std::endl;
      }
      return (List::create(Named("error") = "No checkpoint values provided"));
    }

    List chckpt_values;
    chckpt_values = checkpoint_values;
    List chckpt_estimates = chckpt_values["checkpoint"];

    ELBO_best = chckpt_values["ELBO"];
    niter = chckpt_values["niter"];
    maximum_elapsed_time = chckpt_values["maximum_elapsed_time"];

    for (int i = 0; i < n_parents; i++) {
      List chckpt_each = chckpt_estimates[i];

      P0[i].setApproximatedNu(chckpt_each["nu"]);
      P0[i].setApproximatedLogNu(chckpt_each["log_nu"]);
      P0[i].setApproximatedOmega(chckpt_each["omega"]);
      P0[i].setApproximatedGamma(chckpt_each["gamma"]);
      P0[i].setApproximatedLogGamma(chckpt_each["log_gamma"]);
      P0[i].setApproximatedEtaTilde(chckpt_each["eta_tilde"]);
      P0[i].setApproximatedSigmaTilde(chckpt_each["sigma_tilde"]);
      P0[i].setApproximatedAlpha(chckpt_each["alpha"]);
      P0[i].setApproximatedSigmaShape(chckpt_each["sigma_shape"]);
      P0[i].setApproximatedSigmaScale(chckpt_each["sigma_scale"]);
      P0[i].setApproximatedEtaMean(chckpt_each["eta_mean"]);
      P0[i].setApproximatedEtaVar(chckpt_each["eta_var"]);
      P0[i].setApproximatedRho(chckpt_each["rho"]);
    }
    if (verbose) {
      Rcpp::Rcout << " - Finished loading checkpoint..." << std::endl;
    }
  }
  if (verbose) {
    Rcpp::Rcout << "Finished Generating the Initial Population of size "
                << n_parents << std::endl;
  }

  OptimMember P_current_best = P0[0];
  OptimMember P_previous_best = P0[0];
  OptimMember P_best = P0[0];

  arma::colvec sim_prob_old(N * (N - 1) / 2, arma::fill::ones);
  arma::colvec sim_prob_new(N * (N - 1) / 2, arma::fill::zeros);

  bool elbo_flag = false;
  bool elbo_decreasing = false;
  bool sim_flag = false;
  bool niters_flag = false;
  bool converged = false;
  bool interrupted = false;

  while (!converged) {
    if (verbose) Rcpp::Rcout << "#(Iteration) = " << niter + 1 << std::endl;

    // 2. Update: P_0 -> P_1
    std::vector<OptimMember> P1 =
        update_step(P0, tol_cavi, max_iter_cavi, verbose, n_threads);
    if (verbose) Rcpp::Rcout << "Finished the First Update" << std::endl;

    for (int o = 0; o < n_parents; o++) {
      if (P1[o].getElapsedTime() > maximum_elapsed_time) {
        maximum_elapsed_time = P1[o].getElapsedTime();
      }
    }

    // 3. Crossover: P_1 -> P_2
    std::vector<OptimMember> P2 =
        crossover_step(P1, combination_matrix, n_combination, n_children);
    if (verbose) Rcpp::Rcout << "Finished Crossover Step" << std::endl;

    // 4. Update: P_2 -> P_3
    std::vector<OptimMember> P3 =
        update_step(P2, tol_cavi, max_iter_cavi, verbose, n_threads);
    if (verbose) Rcpp::Rcout << "Finished the Second Update" << std::endl;

    for (int o = 0; o < n_children; o++) {
      if (P3[o].getElapsedTime() > maximum_elapsed_time) {
        maximum_elapsed_time = P3[o].getElapsedTime();
      }
    }

    // 5. Select: P_1 U P_3 -> P_4
    std::vector<OptimMember> P4 = selection_step(P1, P3, n_parents, n_children);
    if (verbose) Rcpp::Rcout << "Finished Selecting Population" << std::endl;

    // 6. Mutate P_4 -> P_5
    std::vector<OptimMember> P5 = mutation_step(P4, n_parents, n_threads);
    if (verbose) Rcpp::Rcout << "Finished Mutation Step" << std::endl;

    // 6. Check convergence
    // Extract the best-performing model
    P_current_best = P4[0];
    ELBO_best.push_back(P_current_best.getELBO());

    arma::mat psm =
        posterior_similarity(P_current_best.getApproximatedNu(), false);
    sim_prob_new = psm.col(2);

    if (niter > 0) {
      if (verbose) {
        Rcpp::Rcout << "Generation #" << niter + 1
                    << ", Current ELBO: " << ELBO_best[niter]
                    << ", Previous ELBO: " << ELBO_best[niter - 1] << std::endl;
      }

      elbo_flag = std::fabs((ELBO_best[niter] - ELBO_best[niter - 1]) /
                            ELBO_best[niter - 1]) < tol_evi;
      elbo_decreasing = ELBO_best[niter] < ELBO_best[niter - 1];
      sim_flag = arma::abs(sim_prob_new - sim_prob_old).max() < tol_evi;
      niters_flag = (niter >= max_iter_evi);

      converged = elbo_flag || elbo_decreasing || sim_flag || niters_flag;
    }

    sim_prob_old = sim_prob_new;
    niter += 1;

    P0 = P5;
    P_previous_best = P_current_best;

    interrupted = ((timer.now() - start_t) / nano) > max_time;
    if (interrupted) {
      std::vector<List> chckpt_vector;

      for (int i = 0; i < n_parents; i++) {
        OptimMember P_tmp = P4[i];

        List chckpt_each = List::create(
            Named("nu") = P_tmp.getApproximatedNu(),
            Named("log_nu") = P_tmp.getApproximatedLogNu(),
            Named("omega") = P_tmp.getApproximatedOmega(),
            Named("gamma") = P_tmp.getApproximatedGamma(),
            Named("log_gamma") = P_tmp.getApproximatedLogGamma(),
            Named("eta_tilde") = P_tmp.getApproximatedEtaTilde(),
            Named("sigma_tilde") = P_tmp.getApproximatedSigmaTilde(),
            Named("alpha") = P_tmp.getApproximatedAlpha(),
            Named("sigma_shape") = P_tmp.getApproximatedSigmaShape(),
            Named("sigma_scale") = P_tmp.getApproximatedSigmaScale(),
            Named("eta_mean") = P_tmp.getApproximatedEtaMean(),
            Named("eta_var") = P_tmp.getApproximatedEtaVar(),
            Named("rho") = P_tmp.getApproximatedRho());

        chckpt_vector.push_back(chckpt_each);
      }

      if (verbose) {
        Rcpp::Rcout << "- Stop Evolutionary CAVI due to the interruption:"
                    << std::endl;
        Rcpp::Rcout << "   Running time reached time limit." << std::endl;
        Rcpp::Rcout
            << "  - Save a checkpoint and resume the optimization later."
            << std::endl;
      }

      return List::create(
          Named("checkpoint") = chckpt_vector, Named("ELBO") = ELBO_best,
          Named("niter") = niter, Named("interruption") = interrupted,
          Named("maximum_elapsed_time") = maximum_elapsed_time,
          Named("elapsed_time") =
              round((timer.now() - start_t) / nano * 10000.0) / 10000.0);
    }

    if (verbose) {
      Rcpp::Rcout << "Finished " << niter << "th iteration... (Elapsed Time: "
                  << round((timer.now() - start_iter) / nano) << " seconds)"
                  << std::endl;
    }
  }

  if (verbose) {
    Rcpp::Rcout << "- Finished Evolutionary CAVI:" << std::endl;
    Rcpp::Rcout << "  - Total Number of Iteration: " << niter << std::endl;
    Rcpp::Rcout << "  - Total Elapsed Time: "
                << round((timer.now() - start_t) / nano) << " seconds"
                << std::endl;
  }

  NumericVector ELBO_output;
  if (elbo_decreasing) {
    P_best = P_previous_best;
    for (int i = 0; i < ELBO_best.size() - 1; i++) {
      ELBO_output.push_back(ELBO_best[i]);
    }
  } else {
    P_best = P_current_best;
    for (int i = 0; i < ELBO_best.size(); i++) {
      ELBO_output.push_back(ELBO_best[i]);
    }
  }

  return List::create(
      Named("nu") = P_best.getApproximatedNu(),
      Named("omega") = P_best.getApproximatedOmega(),
      Named("gamma") = P_best.getApproximatedGamma(),
      Named("eta_tilde") = P_best.getApproximatedEtaTilde(),
      Named("sigma_tilde") = P_best.getApproximatedSigmaTilde(),
      Named("alpha") = P_best.getApproximatedAlpha(),
      Named("sigma_shape") = P_best.getApproximatedSigmaShape(),
      Named("sigma_scale") = P_best.getApproximatedSigmaScale(),
      Named("eta_mean") = P_best.getApproximatedEtaMean(),
      Named("eta_var") = P_best.getApproximatedEtaVar(),
      Named("rho") = P_best.getApproximatedRho(), Named("ELBO") = ELBO_output,
      Named("niter") = niter, Named("interruption") = interrupted,
      Named("maximum_elapsed_time") = maximum_elapsed_time,
      Named("elapsed_time") =
          round((timer.now() - start_t) / nano * 10000.0) / 10000.0);
}

// [[Rcpp::export(name = ".MCMC")]]
List MCMC(arma::field<arma::mat> x, int k, arma::vec n, int N, int p,
          arma::vec discrete_fields, int n_discrete_fields, arma::vec M,
          arma::vec continuous_fields, int n_continuous_fields,
          arma::mat hyper_beta, arma::mat hyper_sigma, arma::vec hyper_phi,
          arma::vec hyper_tau, arma::vec hyper_delta, int n_burnin, int n_gibbs,
          int n_split_merge, bool verbose, double max_time,
          bool custom_initializer, bool use_checkpoint,
          Nullable<List> initial_values, Nullable<List> checkpoint_values) {
  //// Define empty fields to store MCMC samples
  // linkage structure
  arma::field<arma::field<IntegerVector>> lambda_out(n_gibbs);
  // distortion indicator
  arma::field<arma::field<arma::mat>> z_out(n_gibbs);
  // latent true value
  arma::field<arma::mat> y_out(n_gibbs);
  // distortion ratio (all)
  arma::field<NumericVector> beta_out(n_gibbs);
  // probabilities of true values
  arma::field<arma::field<NumericVector>> theta_out(n_gibbs);
  // mean of latent true values
  arma::field<NumericVector> eta_out(n_gibbs);
  // variance of latent true values
  arma::field<NumericVector> sigma_out(n_gibbs);

  // Define empty storage for each iteration
  arma::field<IntegerVector> lambda(k);
  arma::field<arma::mat> z(k);
  arma::mat y(N, p);
  NumericVector beta(p);
  arma::field<NumericVector> theta(n_discrete_fields);
  NumericVector eta(n_continuous_fields);
  NumericVector sigma(n_continuous_fields);

  NumericVector log_odds(p);
  arma::field<NumericVector> log_theta(n_discrete_fields);

  // Define empty storage for each split and merge results
  arma::field<arma::mat> split_merge(5);

  // Set timer to check the computation time
  Timer timer;
  int nano = 1000000000;
  double start_t = timer.now();
  double start_iter;

  bool interrupted = false;

  // internal, hard-coded parameters
  int n_metropolis = 1;
  bool sample_y = false;

  // 1. Initialization
  if (verbose) {
    Rcpp::Rcout << "CHOMPER (MCMC): Start Initialization..." << std::endl;
  }
  // 1.1. Define auxiliary values
  // 1.1.1. Auxiliary index (observations and files) vectors,
  //        which are used for split and merge steps:
  // files: file indices
  // indices: observation indices within each file
  // entire_index: observation indices for the entire dataset
  // n_arma: number of observations in each file (type: arma::vec)
  IntegerVector files = seq(0, k - 1);
  arma::field<IntegerVector> indices(k);
  for (int i = 0; i < k; i++) {
    indices(i) = seq(0, n(i) - 1);
  }
  IntegerVector entire_index = seq(0, N - 1);
  arma::vec n_arma = as<arma::vec>(wrap(n));

  // 1.1.2. Base probability matrix for data:
  //        (i.e., probability of each discrete data point given each level)
  arma::field<arma::mat> x_probability(n_discrete_fields);
  arma::field<arma::mat> log_x_probability(n_discrete_fields);
  // For each discrete field,
  for (int l = 0; l < n_discrete_fields; l++) {
    // Define phi_{l}^{1 / tau_{l}}, softmax-representation parameter
    // We should call hyper_tau(discrete_fields(l)), not hyper_tau(l),
    // because hyper_tau is a vector of length p, not n_discrete_fields
    double log_phi_tau = log(hyper_phi(l)) / hyper_tau(discrete_fields(l));

    // Define probability matrix for each discrete field
    arma::mat each_probability(M(l), M(l));
    arma::mat each_log_probability(M(l), M(l));
    // For each possible level of a categorical variable,
    for (int m_out = 0; m_out < M(l); m_out++) {
      // Initialize a probability vector with ones
      arma::rowvec each_log_level(M(l), arma::fill::zeros);
      // For each possible level of a categorical variable,
      for (int m = 0; m < M(l); m++) {
        // If the |x_{ijl} - m| <= delta_{l},
        if (std::fabs(m_out - m) <= hyper_delta(l)) {
          // Multiply log(phi_{l}^{1 / tau_{l}})
          each_log_level(m) = log_phi_tau;
        }
      }
      // Store the probability vector
      each_probability.row(m_out) = log_softmax(each_log_level, false);
      each_log_probability.row(m_out) = log_softmax(each_log_level, true);
    }
    // Store the probability matrix
    x_probability(l) = each_probability;
    log_x_probability(l) = each_log_probability;
  }

  // 1.1.3. Matrix with indicator of x_{ijl} is in the range of delta_{l},
  //        which is used for generating y_{j'l}
  arma::field<arma::field<arma::mat>> x_in_range(k);
  // For each file,
  for (int i = 0; i < k; i++) {
    // Define an empty field of length n_discrete_fields
    arma::field<arma::mat> file_x_in_range(n_discrete_fields);
    // For each discrete field,
    for (int l = 0; l < n_discrete_fields; l++) {
      // Define an n(i) by M(l) empty matrix to store indicator values
      // 'm'th column of the matrix corresponds to |x_{ijl} - m| <= delta_{l}
      arma::mat each_x_in_range(n(i), M(l), arma::fill::zeros);
      for (int j = 0; j < n(i); j++) {
        for (int m = 0; m < M(l); m++) {
          // If the |x_{ijl} - m| <= delta_{l},
          if (std::fabs(x(i)(j, discrete_fields(l)) - (m + 1)) <=
              hyper_delta(l)) {
            // Set the indicator to one
            each_x_in_range(j, m) = 1;
          }
        }
      }
      file_x_in_range(l) = each_x_in_range;
    }
    x_in_range(i) = file_x_in_range;
  }

  // 1.2. Initialize parameters
  // 1.2.1. Linkage structure and corresponding distortion and true values,
  //        lambda_{ij}, z_{ijl}, y_{j'l}, eta_{l}, and sigma_{l}
  if (custom_initializer) {
    // lambda_{ij} are prespecified based on the domain knowledge,
    // and their corresponding z and y are also given.
    if (initial_values.isNull()) {
      if (verbose) {
        Rcpp::Rcout << "---------------------------------------" << std::endl;
        Rcpp::Rcout << "- ERROR: No initial values provided;   " << std::endl;
        Rcpp::Rcout << "         Discard the current results. -" << std::endl;
        Rcpp::Rcout << "---------------------------------------" << std::endl;
      }
      return (List::create(Named("error") = "No initial values provided"));
    }

    List custom_init_list;
    custom_init_list = initial_values;

    arma::field<IntegerVector> lambda_custom_init = custom_init_list["linkage"];
    arma::field<arma::vec> lambda_argument = custom_init_list["linkage"];

    // Other parameters are initialized based on
    // the given lambda, z, and y.
    for (int i = 0; i < k; i++) {
      lambda(i) = lambda_custom_init(i);
    }

    arma::mat x_append = compile_matrix(x);

    y = init_true_latent(lambda_argument, x_append, k, p, N);
    z = init_distortion(x, y, lambda_argument, k, n, p, discrete_fields,
                        n_discrete_fields, continuous_fields,
                        n_continuous_fields, hyper_delta, hyper_tau);
  } else {
    for (int i = 0; i < k; i++) {
      // lambda_{ij} ~ constant
      // As there is no linkage at the beginning,
      // each x_{ij} is only linked to itself
      // That is, lambda_{ij} = j' where j' is the exact location of a record,
      // when all files are binded horizontally.
      IntegerVector lambda_each(n(i));
      if (i == 0) {
        lambda_each = seq(0, n_arma(i) - 1);
      } else {
        lambda_each =
            seq(sum(n_arma.subvec(0, i - 1)), sum(n_arma.subvec(0, i)) - 1);
      }
      lambda(i) = lambda_each;

      // z_{ijl} ~ Bernoulli(p_{ijl})
      // Distortion indicator is initialized to zero,
      // as we start from the status that there is no linkage among files
      arma::mat z_each(n(i), p, arma::fill::zeros);
      z(i) = z_each;

      // y_{j'l} ~ Multinomial(theta_{j'l}), or Normal(eta_{j'l}, sigma_{j'l})
      // However, we initialize latent records are the same as data
      if (i == 0) {
        y.rows(0, n_arma(i) - 1) = x(i);
      } else {
        y.rows(sum(n_arma.subvec(0, i - 1)), sum(n_arma.subvec(0, i)) - 1) =
            x(i);
      }
    }
  }

  // theta_{l} ~ Dirichlet(mu_{l}), where mu_{l} is rep(1 / M(l), M(l))
  arma::field<NumericVector> mu(n_discrete_fields);
  for (int l = 0; l < n_discrete_fields; l++) {
    // Initialize theta_{l} with E[prior distribution]
    NumericVector mu_each((int)(M(l)), 1.0);
    mu(l) = mu_each;

    theta(l) = mu_each / M(l);

    // log(theta_{l})
    log_theta(l) = log(theta(l));
  }

  for (int l = 0; l < n_continuous_fields; l++) {
    // eta_{l} ~ constant
    // Initialize eta_{l} with the sample mean of the continuous field
    eta(l) = mean(y.col(continuous_fields(l)));

    // sigma_{l} ~ Inverse-Gamma(hyper_sigma_{l1}, hyper_sigma_{l2})
    // Even though sigma_{l} should be initialized with E[prior distribution],
    // because the expectation is not defined for shape < 1,
    // we use the sample variance.
    sigma(l) = var(y.col(continuous_fields(l)));
  }

  // 1.2.2. Distortion Ratio: beta_{l}
  // beta_{l} ~ Beta(hyper_beta_{l1}, hyper_beta_{l2})
  for (int l = 0; l < p; l++) {
    beta(l) = hyper_beta(l, 0) / (hyper_beta(l, 0) + hyper_beta(l, 1));
  }
  // log(beta_{l} / (1.0 - beta_{l}))
  log_odds = log(beta / (1.0 - beta));

  if (use_checkpoint) {
    if (verbose) {
      Rcpp::Rcout << " - Loading checkpoint..." << std::endl;
      Rcpp::Rcout << " - Overriding previous MCMC samples as initial values..."
                  << std::endl;
    }

    if (checkpoint_values.isNull()) {
      if (verbose) {
        Rcpp::Rcout << "----------------------------------------" << std::endl;
        Rcpp::Rcout << "- ERROR: No checkpoint values provided; " << std::endl;
        Rcpp::Rcout << "         Discard the current results.  -" << std::endl;
        Rcpp::Rcout << "----------------------------------------" << std::endl;
      }
      return (List::create(Named("error") = "No checkpoint values provided"));
    }
    List chckpt_values;
    chckpt_values = checkpoint_values;

    arma::field<IntegerVector> lambda_chckpt = chckpt_values["lambda"];
    arma::field<arma::mat> z_chckpt = chckpt_values["z"];
    arma::mat y_chckpt = chckpt_values["y"];
    arma::field<NumericVector> theta_chckpt = chckpt_values["theta"];

    lambda = lambda_chckpt;
    z = z_chckpt;
    y = y_chckpt;

    theta = theta_chckpt;
    for (int l = 0; l < n_discrete_fields; l++) {
      log_theta(l) = log(theta(l));
    }

    eta = chckpt_values["eta"];
    sigma = chckpt_values["sigma"];

    beta = chckpt_values["beta"];
    log_odds = log(beta / (1.0 - beta));
  }

  // 2. MCMC Sampling
  if (verbose) Rcpp::Rcout << "MCMC Starts:" << std::endl;

  // 2.1. Burn-in
  if (verbose) {
    Rcpp::Rcout << " - Burn-in " << n_burnin << " samples..." << std::endl;
  }
  start_iter = timer.now();
  int burnin_term = (int)(n_burnin / 10);
  // Gibbs Sampler within Metropolis-Hastings
  for (int imcmc = 0; imcmc < n_burnin; imcmc++) {
    // Metropolis-Hastings
    for (int imet = 0; imet < n_metropolis; imet++) {
      // Split and Merge
      for (int isim = 0; isim < n_split_merge; isim++) {
        split_merge = do_split_merge(
            files, indices, entire_index, k, n, p, M, x, y, z, lambda, beta,
            theta, eta, sigma, log_odds, log_theta, discrete_fields,
            n_discrete_fields, continuous_fields, n_continuous_fields,
            hyper_phi, hyper_tau, mu, x_probability, log_x_probability,
            x_in_range, sample_y);

        if (split_merge(0)(0, 0) == 1) {
          // Accept Split and Merge Result:
          // i.e., shift Lambda to Lambda', y to y', and z to z'
          for (size_t c = 0; c < split_merge(1).n_rows; c++) {
            // Shift Lambda to Lambda'
            lambda(split_merge(1)(c, 0))(split_merge(1)(c, 1)) =
                split_merge(1)(c, 2);

            // Shift z to z'
            z(split_merge(1)(c, 0)).row(split_merge(1)(c, 1)) =
                split_merge(3).row(c);
          }
          // Shift y to y'
          for (size_t yl = 0; yl < split_merge(4).n_rows; yl++) {
            y.row(split_merge(4)(yl, 0)) = split_merge(2).row(yl);
          }
        }
      }
    }

    // Gibbs Sampler
    // Sampling theta_{l}, eta_{l}, sigma_{l}, and beta_{l}
    for (int l = 0; l < n_discrete_fields; l++) {
      int ldx = discrete_fields(l);

      // theta_{l} | lambda, z, y, x
      theta(l) = rtheta_l(mu(l), x, y, z, ldx, k, M(l));

      // log(theta_{l})
      log_theta(l) = log(theta(l));

      // beta_{l} | z; l = 1, ..., l_1
      beta(ldx) = rbeta_l(hyper_beta(ldx, 0), hyper_beta(ldx, 1), z, ldx, k, N);
    }

    for (int l = 0; l < n_continuous_fields; l++) {
      int ldx = continuous_fields(l);
      // eta_{l} | y, sigma
      eta(l) = reta_l(x, z, y.col(ldx), sigma(l), N, ldx, k);

      // sigma_{l} | y, eta
      sigma(l) =
          rsigma_l(hyper_sigma.row(l), y.col(ldx), eta(l), x, z, ldx, k, N);

      // beta_{l} | z; l = l_1 + 1, ..., p
      beta(ldx) = rbeta_l(hyper_beta(ldx, 0), hyper_beta(ldx, 1), z, ldx, k, N);
    }

    // log(beta_{l} / (1.0 - beta_{l}))
    log_odds = log(beta / (1.0 - beta));

    // Print progress
    if ((((imcmc % burnin_term) == 0) & (imcmc > 0)) || (imcmc == 1)) {
      if (verbose) {
        Rcpp::Rcout << "Finished " << imcmc << "/" << n_burnin
                    << " burnin... (Elapsed Time: "
                    << round((timer.now() - start_iter) / nano) << " seconds)"
                    << std::endl;
      }
    }
  }
  int burnin_et = round((timer.now() - start_iter) / nano);

  // 2.2. Main MCMC
  if (verbose) {
    Rcpp::Rcout << " - Main MCMC " << n_gibbs << " samples..." << std::endl;
  }
  int save_term = (int)(n_gibbs / 10);
  int n_shift = 0;
  int n_sample = 0;
  start_iter = timer.now();
  for (int imcmc = 0; imcmc < n_gibbs; imcmc++) {
    for (int imet = 0; imet < n_metropolis; imet++) {
      // Split and Merge
      for (int isim = 0; isim < n_split_merge; isim++) {
        split_merge = do_split_merge(
            files, indices, entire_index, k, n, p, M, x, y, z, lambda, beta,
            theta, eta, sigma, log_odds, log_theta, discrete_fields,
            n_discrete_fields, continuous_fields, n_continuous_fields,
            hyper_phi, hyper_tau, mu, x_probability, log_x_probability,
            x_in_range, sample_y);

        if (split_merge(0)(0, 0) == 1) {
          // Accept Split and Merge Result:
          // i.e., shift Lambda to Lambda', y to y', and z to z'
          for (size_t c = 0; c < split_merge(1).n_rows; c++) {
            // Shift Lambda to Lambda'
            lambda(split_merge(1)(c, 0))(split_merge(1)(c, 1)) =
                split_merge(1)(c, 2);

            // Shift z to z'
            z(split_merge(1)(c, 0)).row(split_merge(1)(c, 1)) =
                split_merge(3).row(c);
          }
          // Shift y to y'
          for (size_t yl = 0; yl < split_merge(4).n_rows; yl++) {
            y.row(split_merge(4)(yl, 0)) = split_merge(2).row(yl);
          }

          n_shift += 1;
        }
      }
    }

    // Gibbs Sampler
    // Sampling theta_{l}, eta_{l}, sigma_{l}, and beta_{l}
    for (int l = 0; l < n_discrete_fields; l++) {
      int ldx = discrete_fields(l);

      // theta_{l} | lambda, z, y, x
      theta(l) = rtheta_l(mu(l), x, y, z, ldx, k, M(l));

      // log(theta_{l})
      log_theta(l) = log(theta(l));

      // beta_{l} | z; l = 1, ..., l_1
      beta(ldx) = rbeta_l(hyper_beta(ldx, 0), hyper_beta(ldx, 1), z, ldx, k, N);
    }

    for (int l = 0; l < n_continuous_fields; l++) {
      int ldx = continuous_fields(l);
      // eta_{l} | y, sigma
      eta(l) = reta_l(x, z, y.col(ldx), sigma(l), N, ldx, k);

      // sigma_{l} | y, eta
      sigma(l) =
          rsigma_l(hyper_sigma.row(l), y.col(ldx), eta(l), x, z, ldx, k, N);

      // beta_{l} | z; l = l_1 + 1, ..., p
      beta(ldx) = rbeta_l(hyper_beta(ldx, 0), hyper_beta(ldx, 1), z, ldx, k, N);
    }

    // log(beta_{l} / (1.0 - beta_{l}))
    log_odds = log(beta / (1.0 - beta));

    // Copy current samples; due to memory allocation issue
    NumericVector beta_imcmc = clone(beta);
    arma::field<NumericVector> theta_imcmc(n_discrete_fields);
    for (int l = 0; l < n_discrete_fields; l++) {
      theta_imcmc(l) = clone(theta(l));
    }
    NumericVector eta_imcmc = clone(eta);
    NumericVector sigma_imcmc = clone(sigma);
    arma::mat y_imcmc = y;
    arma::field<arma::mat> z_imcmc = z;
    arma::field<IntegerVector> lambda_imcmc(k);
    for (int i = 0; i < k; i++) {
      lambda_imcmc(i) = clone(lambda(i));
    }

    beta_out(imcmc) = beta_imcmc;
    theta_out(imcmc) = theta_imcmc;
    eta_out(imcmc) = eta_imcmc;
    sigma_out(imcmc) = sigma_imcmc;
    y_out(imcmc) = y_imcmc;
    z_out(imcmc) = z_imcmc;
    lambda_out(imcmc) = lambda_imcmc;

    // Print progress
    if (((imcmc % save_term) == 0) & (imcmc > 0)) {
      if (verbose) {
        Rcpp::Rcout << "   - Finished " << imcmc << "/" << n_gibbs
                    << " iteration... (Elapsed Time: "
                    << round((timer.now() - start_iter) / nano) << " seconds)"
                    << std::endl;
      }
    }

    interrupted = ((timer.now() - start_t) / nano) > max_time;
    if (interrupted) {
      break;
    }

    n_sample += 1;
  }
  int main_et = round((timer.now() - start_iter) / nano);
  double total_et = round((timer.now() - start_t) / nano * 10000.0) / 10000.0;

  if (verbose) {
    Rcpp::Rcout << "----------------------------------------" << std::endl;
    Rcpp::Rcout << "Finished CHOMPER (MCMC):" << std::endl;
    Rcpp::Rcout << " - Total Elapsed Time: " << total_et << " seconds"
                << std::endl;
    Rcpp::Rcout << "   - Burn-In: " << burnin_et << " seconds" << std::endl;
    Rcpp::Rcout << "   - Main MCMC: " << main_et << " seconds" << std::endl;
  }

  return (List::create(
      Named("lambda") = lambda_out, Named("z") = z_out, Named("y") = y_out,
      Named("beta") = beta_out, Named("theta") = theta_out,
      Named("eta") = eta_out, Named("sigma") = sigma_out,
      Named("n_shift") = n_shift, Named("n_sample") = n_sample,
      Named("elapsed_time") = total_et, Named("interruption") = interrupted));
}

// [[Rcpp::export(name = ".CoordinateAscentVI")]]
List CoordinateAscentVI(arma::field<arma::mat> x, int k, arma::vec n, int N,
                        int p, arma::vec discrete_fields, int n_discrete_fields,
                        arma::vec M, arma::vec continuous_fields,
                        int n_continuous_fields, arma::mat hyper_beta,
                        arma::mat hyper_sigma, arma::vec hyper_phi,
                        arma::vec hyper_tau, arma::vec hyper_delta,
                        double overlap_prob, double tol_cavi, int max_iter_cavi,
                        bool verbose, double max_time, bool custom_initializer,
                        bool use_checkpoint, Nullable<List> initial_values,
                        Nullable<List> checkpoint_values) {
  Timer timer;
  int nano = 1e9;
  double start_t = timer.now();

  // theta_{l} ~ Dirichlet(mu_{l}), where mu_{l} is rep(1, M(l))
  arma::field<arma::vec> hyper_mu(n_discrete_fields);
  for (int li = 0; li < n_discrete_fields; li++) {
    arma::vec mu_each(M(li), arma::fill::ones);

    hyper_mu(li) = mu_each;
  }

  // Create Initial Data Class
  OptimMember data(x, k, n, N, p, discrete_fields, n_discrete_fields, M,
                   continuous_fields, n_continuous_fields, hyper_beta, hyper_mu,
                   hyper_sigma, hyper_phi, hyper_tau, hyper_delta);

  // Main iterative optimization
  if (verbose) {
    Rcpp::Rcout << "Start Coordinate Ascent Variational Inference..."
                << std::endl;
  }

  double sampling_prob = R::runif(0.0, overlap_prob);
  data.setSamplingProb(sampling_prob);

  // 1. Initialize
  if (custom_initializer) {
    // lambda_{ij} are prespecified based on the domain knowledge.
    if (initial_values.isNull()) {
      if (verbose) {
        Rcpp::Rcout << "---------------------------------------" << std::endl;
        Rcpp::Rcout << "- ERROR: No initial values provided;   " << std::endl;
        Rcpp::Rcout << "         Discard the current results. -" << std::endl;
        Rcpp::Rcout << "---------------------------------------" << std::endl;
      }
      return (List::create(Named("error") = "No initial values provided"));
    }
  }
  data.initialize(custom_initializer, initial_values);

  if (use_checkpoint) {
    if (verbose) {
      Rcpp::Rcout << " - Loading checkpoint..." << std::endl;
      Rcpp::Rcout
          << " - Overriding previous approximations as initial values..."
          << std::endl;
    }

    if (checkpoint_values.isNull()) {
      if (verbose) {
        Rcpp::Rcout << "----------------------------------------" << std::endl;
        Rcpp::Rcout << "- ERROR: No checkpoint values provided; " << std::endl;
        Rcpp::Rcout << "         Discard the current results.  -" << std::endl;
        Rcpp::Rcout << "----------------------------------------" << std::endl;
      }
      return (List::create(Named("error") = "No checkpoint values provided"));
    }

    List chckpt_values;
    chckpt_values = checkpoint_values;

    data.setApproximatedNu(chckpt_values["nu"]);
    data.setApproximatedLogNu(chckpt_values["log_nu"]);
    data.setApproximatedOmega(chckpt_values["omega"]);
    data.setApproximatedGamma(chckpt_values["gamma"]);
    data.setApproximatedLogGamma(chckpt_values["log_gamma"]);
    data.setApproximatedEtaTilde(chckpt_values["eta_tilde"]);
    data.setApproximatedSigmaTilde(chckpt_values["sigma_tilde"]);
    data.setApproximatedAlpha(chckpt_values["alpha"]);
    data.setApproximatedSigmaShape(chckpt_values["sigma_shape"]);
    data.setApproximatedSigmaScale(chckpt_values["sigma_scale"]);
    data.setApproximatedEtaMean(chckpt_values["eta_mean"]);
    data.setApproximatedEtaVar(chckpt_values["eta_var"]);
    data.setApproximatedRho(chckpt_values["rho"]);

    if (verbose) {
      Rcpp::Rcout << " - Finished loading checkpoint..." << std::endl;
    }
  }
  if (verbose) {
    Rcpp::Rcout << "Finished Initialization..." << std::endl;
  }

  data.cavi(tol_cavi, max_iter_cavi, verbose, max_time);

  if (verbose) {
    Rcpp::Rcout << "- Finished Simple CAVI:" << std::endl;
    Rcpp::Rcout << "  - Total Number of Iteration: "
                << data.getNumberOfIterations() << std::endl;
    Rcpp::Rcout << "  - Total Elapsed Time: "
                << round((timer.now() - start_t) / nano) << " seconds"
                << std::endl;
  }

  return List::create(
      Named("nu") = data.getApproximatedNu(),
      Named("omega") = data.getApproximatedOmega(),
      Named("gamma") = data.getApproximatedGamma(),
      Named("eta_tilde") = data.getApproximatedEtaTilde(),
      Named("sigma_tilde") = data.getApproximatedSigmaTilde(),
      Named("alpha") = data.getApproximatedAlpha(),
      Named("sigma_shape") = data.getApproximatedSigmaShape(),
      Named("sigma_scale") = data.getApproximatedSigmaScale(),
      Named("eta_mean") = data.getApproximatedEtaMean(),
      Named("eta_var") = data.getApproximatedEtaVar(),
      Named("rho") = data.getApproximatedRho(), Named("ELBO") = data.getELBO(),
      Named("niter") = data.getNumberOfIterations(),
      Named("interruption") = data.isInterrupted(),
      Named("cavi_elapsed_time") = data.getElapsedTime(),
      Named("elapsed_time") =
          round((timer.now() - start_t) / nano * 10000.0) / 10000.0);
}

// [[Rcpp::export(name = ".DIG")]]
List DIG(arma::field<arma::mat> x, int k, arma::vec n, int N, int p,
         arma::vec discrete_fields, int n_discrete_fields, arma::vec M,
         arma::vec continuous_fields, int n_continuous_fields,
         arma::mat hyper_beta, arma::mat hyper_sigma, arma::vec hyper_phi,
         arma::vec hyper_tau, arma::vec hyper_delta,
         double decaying_upper_bound, int n_burnin, int n_gibbs, int batch_size,
         int n_epochs, double max_time, bool batch_update) {
  //// Define empty fields to store MCMC samples
  // linkage structure
  arma::field<arma::field<IntegerVector>> lambda_out(n_gibbs);
  // distortion indicator
  arma::field<arma::field<arma::mat>> z_out(n_gibbs);
  // latent true value
  arma::field<arma::mat> y_out(n_gibbs);
  // distortion ratio (all)
  arma::field<NumericVector> beta_out(n_gibbs);
  // probabilities of true values
  arma::field<arma::field<NumericVector>> theta_out(n_gibbs);
  // mean of latent true values
  arma::field<NumericVector> eta_out(n_gibbs);
  // variance of latent true values
  arma::field<NumericVector> sigma_out(n_gibbs);

  // Define empty storage for each iteration
  arma::field<IntegerVector> lambda(k);
  arma::field<arma::mat> z(k);
  arma::mat y(N, p);
  NumericVector beta(p);
  arma::field<NumericVector> theta(n_discrete_fields);
  NumericVector eta(n_continuous_fields);
  NumericVector sigma(n_continuous_fields);

  NumericVector log_odds(p);
  arma::field<NumericVector> log_theta(n_discrete_fields);

  // Define xi's for phase transition
  int total_iteration = n_burnin + n_gibbs;
  int xi1 = std::floor(total_iteration * 0.25);
  int xi2 = std::floor(total_iteration * 0.5);

  // Define probabilities for calculating discomfort and actual discomfort.
  arma::vec discomfort_probability(N);
  arma::vec discomfort(N);

  // Initialize decaying parameter
  double decaying_parameter = decaying_upper_bound;

  // Auxiliary index matrix for sampling
  arma::mat sampling_index = create_index_matrix(k, n);

  // Set timer to check the computation time
  Timer timer;
  int nano = 1000000000;
  double start_t = timer.now();
  double start_iter;

  bool interrupted = false;

  // 1. Initialization
  Rcpp::Rcout << "CHOMPER (DIG): Start Initialization..." << std::endl;
  // 1.1. Define auxiliary values
  // 1.1.1. Auxiliary index (observations and files) vectors,
  //        which are used for split and merge steps:
  // files: file indices
  // indices: observation indices within each file
  // entire_index: observation indices for the entire dataset
  // n_arma: number of observations in each file (type: arma::vec)
  IntegerVector files = seq(0, k - 1);
  arma::field<IntegerVector> indices(k);
  for (int i = 0; i < k; i++) {
    indices(i) = seq(0, n(i) - 1);
  }
  IntegerVector entire_index = seq(0, N - 1);
  arma::vec n_arma = as<arma::vec>(wrap(n));

  // 1.1.2. Base probability matrix for data:
  //        (i.e., probability of each discrete data point given each level)
  arma::field<arma::mat> x_probability(n_discrete_fields);
  arma::field<arma::mat> log_x_probability(n_discrete_fields);
  arma::vec log_phi_tau(n_discrete_fields);
  // For each discrete field,
  for (int l = 0; l < n_discrete_fields; l++) {
    // Define phi_{l}^{1 / tau_{l}}, softmax-representation parameter
    // We should call hyper_tau(discrete_fields(l)), not hyper_tau(l),
    // because hyper_tau is a vector of length p, not n_discrete_fields
    log_phi_tau(l) = log(hyper_phi(l)) / hyper_tau(discrete_fields(l));

    // Define probability matrix for each discrete field
    arma::mat each_probability(M(l), M(l));
    arma::mat each_log_probability(M(l), M(l));
    // For each possible level of a categorical variable,
    for (int m_out = 0; m_out < M(l); m_out++) {
      // Initialize a probability vector with ones
      arma::rowvec each_log_level(M(l), arma::fill::zeros);
      // For each possible level of a categorical variable,
      for (int m = 0; m < M(l); m++) {
        // If the |x_{ijl} - m| <= delta_{l},
        if (abs(m_out - m) <= hyper_delta(l)) {
          // Multiply log(phi_{l}^{1 / tau_{l}})
          each_log_level(m) = log_phi_tau(l);
        }
      }
      // Store the probability vector
      each_probability.row(m_out) = log_softmax(each_log_level, false);
      each_log_probability.row(m_out) = log_softmax(each_log_level, true);
    }
    // Store the probability matrix
    x_probability(l) = each_probability;
    log_x_probability(l) = each_log_probability;
  }

  // 1.1.3. Matrix with indicator of x_{ijl} is in the range of delta_{l},
  //        which is used for generating y_{j'l}
  arma::field<arma::field<arma::mat>> x_in_range(k);
  // For each file,
  for (int i = 0; i < k; i++) {
    // Define an empty field of length n_discrete_fields
    arma::field<arma::mat> file_x_in_range(n_discrete_fields);
    // For each discrete field,
    for (int l = 0; l < n_discrete_fields; l++) {
      // Define an n(i) by M(l) empty matrix to store indicator values
      // 'm'th column of the matrix corresponds to |x_{ijl} - m| <= delta_{l}
      arma::mat each_x_in_range(n(i), M(l), arma::fill::zeros);
      for (int j = 0; j < n(i); j++) {
        for (int m = 0; m < M(l); m++) {
          // If the |x_{ijl} - m| <= delta_{l},
          if (abs(x(i)(j, discrete_fields(l)) - (m + 1)) <= hyper_delta(l)) {
            // Set the indicator to one
            each_x_in_range(j, m) = 1;
          }
        }
      }
      file_x_in_range(l) = each_x_in_range;
    }
    x_in_range(i) = file_x_in_range;
  }

  // 1.2. Initialize parameters
  // 1.2.1. Linkage structure and corresponding distortion and true values,
  //        lambda_{ij}, z_{ijl}, y_{j'l}, eta_{l}, and sigma_{l}
  for (int i = 0; i < k; i++) {
    // lambda_{ij} ~ constant
    // As there is no linkage at the beginning,
    // each x_{ij} is only linked to itself
    // That is, lambda_{ij} = j' where j' is the exact location of a record,
    // when all files are binded horizontally.
    IntegerVector lambda_each(n(i));
    if (i == 0) {
      lambda_each = seq(0, n_arma(i) - 1);
    } else {
      lambda_each =
          seq(sum(n_arma.subvec(0, i - 1)), sum(n_arma.subvec(0, i)) - 1);
    }
    lambda(i) = lambda_each;

    // z_{ijl} ~ Bernoulli(p_{ijl})
    // Distortion indicator is initialized to zero,
    // as we start from the status that there is no linkage among files
    arma::mat z_each(n(i), p, arma::fill::zeros);
    z(i) = z_each;

    // y_{j'l} ~ Multinomial(theta_{j'l}), or Normal(eta_{j'l}, sigma_{j'l})
    // However, we initialize latent records are the same as data
    if (i == 0) {
      y.rows(0, n_arma(i) - 1) = x(i);
    } else {
      y.rows(sum(n_arma.subvec(0, i - 1)), sum(n_arma.subvec(0, i)) - 1) = x(i);
    }
  }

  // theta_{l} ~ Dirichlet(mu_{l}), where mu_{l} is rep(1 / M(l), M(l))
  arma::field<NumericVector> mu(n_discrete_fields);
  for (int l = 0; l < n_discrete_fields; l++) {
    // Initialize theta_{l} with E[prior distribution]
    NumericVector mu_each((int)(M(l)), 1.0);
    mu(l) = mu_each;

    theta(l) = mu_each / M(l);

    // log(theta_{l})
    log_theta(l) = log(theta(l));
  }

  for (int l = 0; l < n_continuous_fields; l++) {
    // eta_{l} ~ constant
    // Initialize eta_{l} with the sample mean of the continuous field
    eta(l) = mean(y.col(continuous_fields(l)));

    // sigma_{l} ~ Inverse-Gamma(hyper_sigma_{l1}, hyper_sigma_{l2})
    // Even though sigma_{l} should be initialized with E[prior distribution],
    // because the expectation is not defined for shape < 1,
    // we use the sample variance.
    sigma(l) = var(y.col(continuous_fields(l)));
  }

  // 1.2.2. Distortion Ratio: beta_{l}
  // beta_{l} ~ Beta(hyper_beta_{l1}, hyper_beta_{l2})
  for (int l = 0; l < p; l++) {
    beta(l) = hyper_beta(l, 0) / (hyper_beta(l, 0) + hyper_beta(l, 1));
  }
  // log(beta_{l} / (1.0 - beta_{l}))
  log_odds = log(beta / (1.0 - beta));

  // 1.3. Initialize allocation matrix
  // Initial allocation matrix is calculated with the initial parameters.
  // Before the initialization, we should define initial nu.
  // 1.3.1. Initialize nu
  // nu is initialized under the assumption
  // that each record is equally likely to be linked to any other record.
  // That is, nu_{ij} = 1 / N for all i, j.
  // TODO-DIG-1--------------------------------------------------------------*//
  // Compare with using initial lambda, assigning large values
  // to the location of initial lambda.
  //*------------------------------------------------------------------------*//
  arma::field<arma::mat> nu(k);
  for (int i = 0; i < k; i++) {
    nu(i) = arma::mat(n(i), N, arma::fill::ones);
    nu(i) /= static_cast<double>(N);
  }

  // 1.3.2. Update (initialize, actually) allocation matrix
  arma::field<arma::mat> allocation_matrix = update_allocation_matrix(
      x, y, z, lambda, nu, log_theta, eta, sigma, log_x_probability, hyper_tau,
      discrete_fields, n_discrete_fields, continuous_fields,
      n_continuous_fields, k, n, N);

  // 2. MCMC Sampling
  int weight_transition_cycle = n_epochs * N / batch_size;
  if (weight_transition_cycle > (n_burnin + n_gibbs)) {
    Rcpp::Rcout
        << "Warning: You need more samples to perform DIG properly. "
           "The weight transition cycle is longer than the total MCMC samples."
        << std::endl;
    Rcpp::Rcout << "         The weight transition is disabled." << std::endl;
  }

  Rcpp::Rcout << "MCMC Starts:" << std::endl;
  // 2.1. Burn-in
  Rcpp::Rcout << " - Burn-in " << n_burnin << " samples..." << std::endl;
  start_iter = timer.now();
  int burnin_term = (int)(n_burnin / 10);

  double batch_size_double = static_cast<double>(batch_size);
  int total_mcmc = 0;
  bool tanh_weight = true;
  int update_cycle = 3;

  double f_weight = 0.0;
  double g_weight = 0.0;
  arma::vec sampling_weights(N, arma::fill::zeros);
  // Discomfort-Informed Adaptive Gibbs Sampler
  for (int imcmc = 0; imcmc < n_burnin; imcmc++) {
    if ((total_mcmc > xi1) & (total_mcmc <= xi2)) {
      update_cycle = 6;
    } else if (total_mcmc > xi2) {
      update_cycle = 10;
    }

    if ((imcmc % update_cycle) == 0) {
      // Update allocation matrix
      // First, calculate `nu` using the current samples.
      nu = update_nu(x, y, z, discrete_fields, n_discrete_fields,
                     continuous_fields, n_continuous_fields, k, n, N,
                     log_phi_tau, hyper_delta, hyper_tau);

      // Then, update allocation matrix using the current samples
      allocation_matrix = update_allocation_matrix(
          x, y, z, lambda, nu, log_theta, eta, sigma, log_x_probability,
          hyper_tau, discrete_fields, n_discrete_fields, continuous_fields,
          n_continuous_fields, k, n, N);
    }

    // Every iteration
    discomfort_probability =
        calculate_discomfort_probability(allocation_matrix, lambda, k);
    discomfort = arma::exp(-decaying_parameter * discomfort_probability);

    if (tanh_weight) {
      decaying_parameter = optimize_decaying_parameter(
          discomfort_probability, batch_size_double, decaying_upper_bound);
      f_weight = f_tanh(total_mcmc, n_epochs);
      g_weight = g_tanh(total_mcmc, n_epochs);
    } else {
      decaying_parameter = 1.0;
      f_weight = f_poly(total_mcmc, n_epochs);
      g_weight = g_poly(total_mcmc, n_epochs);
    }

    sampling_weights = f_weight * sampling_weights + g_weight * discomfort;

    arma::mat sampled_index =
        sample_index_matrix(sampling_index, sampling_weights, batch_size, N);
    for (int cdx = 0; cdx < batch_size; cdx++) {
      int i = sampled_index(cdx, 0);
      int j = sampled_index(cdx, 1);

      arma::rowvec weights_ij = allocation_matrix(i).row(j);

      // Update linkage structure lambda_{ij} using allocation matrix
      lambda(i)(j) =
          sample(entire_index, 1, false,
                 NumericVector(weights_ij.begin(), weights_ij.end()))[0];
    }

    // Sample the rest of the parameters
    // y, z: the order matters,
    // because the updated lambda affects the latent records,
    // and finally the new distortion indicator should be updated
    // based on the updated latent records.
    // However, as batch_size lambdas are updated,
    // we need to sample y_{jprime} for all jprime from new lambda assignments.
    if (batch_update) {
      for (int cdx = 0; cdx < batch_size; cdx++) {
        int i = sampled_index(cdx, 0);
        int j = sampled_index(cdx, 1);
        int jprime = lambda(i)(j);

        y.row(jprime) = update_latent_record(
            hyper_phi, hyper_tau, x, lambda, z, theta, eta, sigma, jprime, k, n,
            M, p, discrete_fields, n_discrete_fields, continuous_fields,
            n_continuous_fields, x_in_range);

        z(i).row(j) = update_distortion(
            x(i).row(j), y.row(jprime), beta, theta, eta, sigma, p,
            discrete_fields, n_discrete_fields, continuous_fields,
            n_continuous_fields, hyper_tau, x_probability);
      }
    } else {
      // Sampling y_{jprime} for all jprime = 1, ..., N
      for (int jprime = 0; jprime < N; jprime++) {
        y.row(jprime) = update_latent_record(
            hyper_phi, hyper_tau, x, lambda, z, theta, eta, sigma, jprime, k, n,
            M, p, discrete_fields, n_discrete_fields, continuous_fields,
            n_continuous_fields, x_in_range);
      }

      // Sampling z_{ij} for all i = 1, ..., k and j = 1, ..., n(i)
      for (int i = 0; i < k; i++) {
        for (int j = 0; j < n(i); j++) {
          z(i).row(j) = update_distortion(
              x(i).row(j), y.row(lambda(i)(j)), beta, theta, eta, sigma, p,
              discrete_fields, n_discrete_fields, continuous_fields,
              n_continuous_fields, hyper_tau, x_probability);
        }
      }
    }

    // Sampling theta_{l}, eta_{l}, sigma_{l}, and beta_{l}
    for (int l = 0; l < n_discrete_fields; l++) {
      int ldx = discrete_fields(l);

      // theta_{l} | lambda, z, y, x
      theta(l) = rtheta_l(mu(l), x, y, z, ldx, k, M(l));

      // log(theta_{l})
      log_theta(l) = log(theta(l));

      // beta_{l} | z; l = 1, ..., l_1
      beta(ldx) = rbeta_l(hyper_beta(ldx, 0), hyper_beta(ldx, 1), z, ldx, k, N);
    }

    for (int l = 0; l < n_continuous_fields; l++) {
      int ldx = continuous_fields(l);
      // eta_{l} | y, sigma
      eta(l) = reta_l(x, z, y.col(ldx), sigma(l), N, ldx, k);

      // sigma_{l} | y, eta
      sigma(l) =
          rsigma_l(hyper_sigma.row(l), y.col(ldx), eta(l), x, z, ldx, k, N);

      // beta_{l} | z; l = l_1 + 1, ..., p
      beta(ldx) = rbeta_l(hyper_beta(ldx, 0), hyper_beta(ldx, 1), z, ldx, k, N);
    }

    // log(beta_{l} / (1.0 - beta_{l}))
    log_odds = log(beta / (1.0 - beta));

    total_mcmc += 1;

    if (total_mcmc > weight_transition_cycle) {
      tanh_weight = false;
    }

    // Print progress
    if ((((imcmc % burnin_term) == 0) & (imcmc > 0)) || (imcmc == 1)) {
      Rcpp::Rcout << "Finished " << imcmc << "/" << n_burnin
                  << " burnin... (Elapsed Time: "
                  << round((timer.now() - start_iter) / nano) << " seconds)"
                  << std::endl;
    }
  }
  int burnin_et = round((timer.now() - start_iter) / nano);

  // 2.2. Main MCMC
  Rcpp::Rcout << " - Main MCMC " << n_gibbs << " samples..." << std::endl;
  int save_term = (int)(n_gibbs / 10);
  int n_sample = 0;
  start_iter = timer.now();
  for (int imcmc = 0; imcmc < n_gibbs; imcmc++) {
    if ((total_mcmc > xi1) & (total_mcmc <= xi2)) {
      update_cycle = 6;
    } else if (total_mcmc > xi2) {
      update_cycle = 10;
    }

    if ((imcmc % update_cycle) == 0) {
      // Update allocation matrix
      // First, calculate `nu` using the current samples.
      nu = update_nu(x, y, z, discrete_fields, n_discrete_fields,
                     continuous_fields, n_continuous_fields, k, n, N,
                     log_phi_tau, hyper_delta, hyper_tau);

      // Then, update allocation matrix using the current samples
      allocation_matrix = update_allocation_matrix(
          x, y, z, lambda, nu, log_theta, eta, sigma, log_x_probability,
          hyper_tau, discrete_fields, n_discrete_fields, continuous_fields,
          n_continuous_fields, k, n, N);
    }

    // Every iteration
    discomfort_probability =
        calculate_discomfort_probability(allocation_matrix, lambda, k);
    discomfort = arma::exp(-decaying_parameter * discomfort_probability);

    if (tanh_weight) {
      decaying_parameter = optimize_decaying_parameter(
          discomfort_probability, batch_size_double, decaying_upper_bound);
      f_weight = f_tanh(total_mcmc, n_epochs);
      g_weight = g_tanh(total_mcmc, n_epochs);
    } else {
      decaying_parameter = 1.0;
      f_weight = f_poly(total_mcmc, n_epochs);
      g_weight = g_poly(total_mcmc, n_epochs);
    }

    sampling_weights = f_weight * sampling_weights + g_weight * discomfort;

    arma::mat sampled_index =
        sample_index_matrix(sampling_index, sampling_weights, batch_size, N);
    for (int cdx = 0; cdx < batch_size; cdx++) {
      int i = sampled_index(cdx, 0);
      int j = sampled_index(cdx, 1);

      arma::rowvec weights_ij = allocation_matrix(i).row(j);

      // Update linkage structure lambda_{ij} using allocation matrix
      lambda(i)(j) =
          sample(entire_index, 1, false,
                 NumericVector(weights_ij.begin(), weights_ij.end()))[0];
    }

    // Sample the rest of the parameters
    // y, z: the order matters,
    // because the updated lambda affects the latent records,
    // and finally the new distortion indicator should be updated
    // based on the updated latent records.
    // However, as batch_size lambdas are updated,
    // we need to sample y_{jprime} for all jprime from new lambda assignments.
    if (batch_update) {
      for (int cdx = 0; cdx < batch_size; cdx++) {
        int i = sampled_index(cdx, 0);
        int j = sampled_index(cdx, 1);
        int jprime = lambda(i)(j);

        y.row(jprime) = update_latent_record(
            hyper_phi, hyper_tau, x, lambda, z, theta, eta, sigma, jprime, k, n,
            M, p, discrete_fields, n_discrete_fields, continuous_fields,
            n_continuous_fields, x_in_range);

        z(i).row(j) = update_distortion(
            x(i).row(j), y.row(jprime), beta, theta, eta, sigma, p,
            discrete_fields, n_discrete_fields, continuous_fields,
            n_continuous_fields, hyper_tau, x_probability);
      }
    } else {
      // Sampling y_{jprime} for all jprime = 1, ..., N
      for (int jprime = 0; jprime < N; jprime++) {
        y.row(jprime) = update_latent_record(
            hyper_phi, hyper_tau, x, lambda, z, theta, eta, sigma, jprime, k, n,
            M, p, discrete_fields, n_discrete_fields, continuous_fields,
            n_continuous_fields, x_in_range);
      }

      // Sampling z_{ij} for all i = 1, ..., k and j = 1, ..., n(i)
      for (int i = 0; i < k; i++) {
        for (int j = 0; j < n(i); j++) {
          z(i).row(j) = update_distortion(
              x(i).row(j), y.row(lambda(i)(j)), beta, theta, eta, sigma, p,
              discrete_fields, n_discrete_fields, continuous_fields,
              n_continuous_fields, hyper_tau, x_probability);
        }
      }
    }

    // Sampling theta_{l}, eta_{l}, sigma_{l}, and beta_{l}
    for (int l = 0; l < n_discrete_fields; l++) {
      int ldx = discrete_fields(l);

      // theta_{l} | lambda, z, y, x
      theta(l) = rtheta_l(mu(l), x, y, z, ldx, k, M(l));

      // log(theta_{l})
      log_theta(l) = log(theta(l));

      // beta_{l} | z; l = 1, ..., l_1
      beta(ldx) = rbeta_l(hyper_beta(ldx, 0), hyper_beta(ldx, 1), z, ldx, k, N);
    }

    for (int l = 0; l < n_continuous_fields; l++) {
      int ldx = continuous_fields(l);
      // eta_{l} | y, sigma
      eta(l) = reta_l(x, z, y.col(ldx), sigma(l), N, ldx, k);

      // sigma_{l} | y, eta
      sigma(l) =
          rsigma_l(hyper_sigma.row(l), y.col(ldx), eta(l), x, z, ldx, k, N);

      // beta_{l} | z; l = l_1 + 1, ..., p
      beta(ldx) = rbeta_l(hyper_beta(ldx, 0), hyper_beta(ldx, 1), z, ldx, k, N);
    }

    // log(beta_{l} / (1.0 - beta_{l}))
    log_odds = log(beta / (1.0 - beta));

    total_mcmc += 1;

    if (total_mcmc > weight_transition_cycle) {
      tanh_weight = false;
    }

    // Copy current samples; due to memory allocation issue
    NumericVector beta_imcmc = clone(beta);
    arma::field<NumericVector> theta_imcmc(n_discrete_fields);
    for (int l = 0; l < n_discrete_fields; l++) {
      theta_imcmc(l) = clone(theta(l));
    }
    NumericVector eta_imcmc = clone(eta);
    NumericVector sigma_imcmc = clone(sigma);
    arma::mat y_imcmc = y;
    arma::field<arma::mat> z_imcmc = z;
    arma::field<IntegerVector> lambda_imcmc(k);
    for (int i = 0; i < k; i++) {
      lambda_imcmc(i) = clone(lambda(i));
    }

    beta_out(imcmc) = beta_imcmc;
    theta_out(imcmc) = theta_imcmc;
    eta_out(imcmc) = eta_imcmc;
    sigma_out(imcmc) = sigma_imcmc;
    y_out(imcmc) = y_imcmc;
    z_out(imcmc) = z_imcmc;
    lambda_out(imcmc) = lambda_imcmc;

    // Print progress
    if (((imcmc % save_term) == 0) & (imcmc > 0)) {
      Rcpp::Rcout << "   - Finished " << imcmc << "/" << n_gibbs
                  << " iteration... (Elapsed Time: "
                  << round((timer.now() - start_iter) / nano) << " seconds)"
                  << std::endl;
    }

    interrupted = ((timer.now() - start_t) / nano) > max_time;
    if (interrupted) {
      break;
    }

    n_sample += 1;
  }
  int main_et = round((timer.now() - start_iter) / nano);
  double total_et = round((timer.now() - start_t) / nano * 10000.0) / 10000.0;

  Rcpp::Rcout << "----------------------------------------" << std::endl;
  Rcpp::Rcout << "Finished CHOMPER (DIG):" << std::endl;
  Rcpp::Rcout << " - Total Elapsed Time: " << total_et << " seconds"
              << std::endl;
  Rcpp::Rcout << "   - Burn-In: " << burnin_et << " seconds" << std::endl;
  Rcpp::Rcout << "   - Main MCMC: " << main_et << " seconds" << std::endl;

  return (List::create(
      Named("lambda") = lambda_out, Named("z") = z_out, Named("y") = y_out,
      Named("beta") = beta_out, Named("theta") = theta_out,
      Named("eta") = eta_out, Named("sigma") = sigma_out,
      Named("n_sample") = n_sample, Named("transition") = !tanh_weight,
      Named("transition_point") = weight_transition_cycle,
      Named("elapsed_time") = total_et, Named("interruption") = interrupted));
}

// Calculate the posterior similarity matrix
// using the parameters 'nu' of approximated posterior
// obtained from VI (or EVIL) CHOMPER
// It returns either the symmetric matrix
// or the long-form matrix with the indexes of pairs
//
// @param probs_field a field of matrices with posterior probabilities
// @return result posterior similarity of all possible pairs
// [[Rcpp::export(name = ".psm_vi")]]
arma::mat psm_vi(arma::field<arma::mat> probs_field) {
  arma::mat probs = compile_matrix(probs_field);
  int n = probs.n_rows;

  arma::mat result = probs * probs.t();
  for (int i = 0; i < n; i++) {
    result(i, i) = 1.0;
  }

  return result;
}

// Calculate the posterior similarity matrix
// using the MCMC samples of 'lambda' from MCMC-CHOMPER
// It returns the symmetric matrix
//
// @param samples an N by nmcmc matrix with MCMC samples
// @return result posterior similarity of all possible pairs
// [[Rcpp::export(name = ".psm_mcmc")]]
arma::mat psm_mcmc(arma::mat samples) {
  int N = samples.n_cols;
  int nmcmc = samples.n_rows;
  double dmcmc = double(nmcmc);

  arma::mat result(N, N);
  for (int i = 0; i < N; i++) {
    for (int j = i; j < N; j++) {
      int count = 0;
      for (int k = 0; k < nmcmc; k++) {
        if (samples(k, i) == samples(k, j)) {
          count++;
        }
      }
      double sim = count / dmcmc;
      result(i, j) = sim;
      result(j, i) = sim;
    }
  }
  return result;
}

// Flatten the posterior samples (lambda) into a matrix
//
// @param samples a field of fields of IntegerVectors
// @param k number of files
// @param N total number of records, sum(n)
// @return result flattened posterior samples
// [[Rcpp::export(name = ".flatten_posterior_samples")]]
arma::imat flatten_posterior_samples(
    arma::field<arma::field<IntegerVector>> samples, int k, int N) {
  int n_sample = samples.n_elem;
  arma::imat result(n_sample, N);

  for (int i = 0; i < n_sample; i++) {
    arma::field<IntegerVector> sample_i = samples(i);
    int col_idx = 0;
    for (int j = 0; j < k; j++) {
      IntegerVector sample_ij = sample_i(j);
      for (int cdx = 0; cdx < sample_ij.size(); cdx++) {
        result(i, col_idx++) = sample_ij(cdx);
      }
    }
  }

  return result;
}

// Evaluate the performance of the linkage structure estimation
//
// @param estimation estimated linkage structure (lambda)
// @param truth true linkage structure
// @param N number of total records
// @param return_matrix bool, if true, return the matrix of linkage structure
// @return performance evaluation
// [[Rcpp::export(name = ".evaluate_performance")]]
List evaluate_performance(arma::field<IntegerVector> estimation,
                          arma::field<IntegerVector> truth, int N,
                          bool return_matrix) {
  arma::mat combination_matrix = create_combination(N);
  IntegerMatrix combination_result(combination_matrix.n_rows, 2);
  combination_result.fill(0);

  for (int i = 1; i <= N; i++) {
    IntegerVector diff_i(1);
    diff_i(0) = i;
    int iloc = N * (i - 1) - i * (i + 1) / 2 - 1;
    if (estimation(i - 1).length() > 1) {
      IntegerVector j_vec = sort_unique(setdiff(estimation(i - 1), diff_i));
      for (int j = 0; j < j_vec.length(); j++) {
        if (j_vec(j) < i) {
          continue;
        } else {
          combination_result(iloc + j_vec(j), 0) = 1;
        }
      }
    }
    if (truth(i - 1).length() > 1) {
      IntegerVector t_vec = sort_unique(setdiff(truth(i - 1), diff_i));
      for (int t = 0; t < t_vec.length(); t++) {
        if (t_vec(t) < i) {
          continue;
        } else {
          combination_result(iloc + t_vec(t), 1) = 1;
        }
      }
    }
  }

  double tp = 0;
  double fp = 0;
  for (unsigned int i = 0; i < combination_matrix.n_rows; i++) {
    if (combination_result(i, 1) == 0) {
      if (combination_result(i, 0) == 1) {
        // Truth: Not Linked, Estimation: Linked
        // False Positive
        fp += 1;
      }
    } else if (combination_result(i, 1) == 1) {
      if (combination_result(i, 0) == 1) {
        // Truth: Linked, Estimation: Linked
        // True Positive
        tp += 1;
      }
    }
  }

  double tn = 0;
  double fn = 0;
  for (int i = 0; i < N; i++) {
    if ((truth(i).length() == 1) && (estimation(i).length() == 1)) {
      // Truth: Not Linked, Estimation: Not Linked
      // True Negative
      tn += 1;
    } else if ((truth(i).length() > 1) && (estimation(i).length() == 1)) {
      // Truth: Linked, Estimation: Not Linked
      // False Negative
      fn += 1;
    }
  }

  if (return_matrix) {
    arma::mat out(combination_matrix.n_rows, 4);

    out.col(0) = combination_matrix.col(0);
    out.col(1) = combination_matrix.col(1);
    out.col(2) = as<arma::colvec>(wrap(combination_result(_, 0)));
    out.col(3) = as<arma::colvec>(wrap(combination_result(_, 1)));

    return List::create(Named("tp") = tp, Named("tn") = tn, Named("fp") = fp,
                        Named("fn") = fn, Named("fpr") = fp / (fp + tn),
                        Named("fnr") = fn / (tp + fn),
                        Named("combination") = out);
  } else {
    return List::create(Named("tp") = tp, Named("tn") = tn, Named("fp") = fp,
                        Named("fn") = fn, Named("fpr") = fp / (fp + tn),
                        Named("fnr") = fn / (tp + fn));
  }
}
