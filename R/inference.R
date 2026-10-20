#' @title CHOMPER with Evolutionary Variational Inference for Record Linkage
#' @description
#' Fit the CHOMPER model with Evolutionary Variational Inference for record linkage (EVIL) to estimate the linkage structure across multiple datasets.
#' It returns the approximate variational factors of the linkage structure that maximize the evidence lower bound (ELBO) and other parameters of the CHOMPER model.
#'
#' @param x A list of data frames, each representing a dataset.
#' @param k The number of datasets to be linked.
#' @param n The number of rows in each dataset (vector of length k).
#' @param N The number of columns in each dataset.
#' @param p The number of fields in each dataset.
#' @param M The number of categories for each discrete field (vector of length of discrete fields).
#' @param discrete_fields The indexes of the discrete fields (1-based index).
#' @param continuous_fields The indexes of the continuous fields (1-based index).
#' @param hyper_beta The hyperparameters for the beta distribution (matrix of size p x 2).
#' @param hyper_phi The hyperparameters for softmax representation (vector of length of discrete fields).
#' @param hyper_tau The temperature parameter (vector of length of discrete fields).
#' @param hyper_epsilon_discrete The range parameter for the comprehensive hit of discrete fields (vector of length of discrete fields).
#' @param hyper_epsilon_continuous The range parameter for the comprehensive hit of continuous fields (vector of length of continuous fields).
#' @param hyper_sigma The hyperparameters for the Inverse Gamma distribution (matrix of size length of continuous fields x 2).
#' @param overlap_prob The presumed probability of overlap across the datasets.
#' @param n_parents The number of parents for a generation.
#' @param n_children The number of children for the next generation.
#' @param tol_cavi The tolerance for the coordinate ascent variational inference for the convergence.
#' @param max_iter_cavi The maximum number of iterations for the coordinate ascent variational inference.
#' @param tol_evi The tolerance for the evolutionary variational inference for the convergence.
#' @param max_iter_evi The maximum number of iterations for the evolutionary variational inference.
#' @param n_threads The number of threads for parallel computation.
#' @param max_time The maximum time limit for the execution in seconds.
#' @param custom_initializer Whether to use a custom initializer for the initial values.
#' @param use_checkpoint Whether to use a checkpoint.
#' @param initial_values The initial values for the parameters (optional).
#' @param checkpoint_values The checkpoint values for the parameters (optional).
#' @param verbose_internal Whether to print the internal C++ messages (TRUE: print, FALSE: not print).
#'
#' @return A list of the approximated parameters of the variational factors and other information containing:
#' \itemize{
#'   \item \code{nu}: A list of parameter matrices for the approximate multinomial posterior of the linkage structure.
#'   \item \code{omega}: A matrix of parameter vectors for the approximate beta posterior of the distortion ratio.
#'   \item \code{rho}: A list of parameter matrices for the approximate Bernoulli posterior of the distortion indicators.
#'   \item \code{gamma}: A list of parameter matrices for the approximate multinomial posterior of discrete true latent values.
#'   \item \code{alpha}: A list of parameter vectors for the approximate Dirichlet posterior (theta) of the discrete true latent values.
#'   \item \code{eta_tilde}: A matrix of parameter vectors for the mean of the approximate normal posterior of continuous true latent values.
#'   \item \code{eta_mean}: A vector of mean parameters for the approximate normal posterior (eta_tilde) of the continuous true latent values.
#'   \item \code{eta_var}: A vector of variance parameters for the approximate normal posterior (eta_tilde) of the continuous true latent values.
#'   \item \code{sigma_tilde}: A matrix of parameter vectors for the variance of the approximate normal posterior of continuous true latent values.
#'   \item \code{sigma_shape}: A vector of shape parameters for the approximate inverse gamma posterior (sigma_tilde) of the continuous true latent values.
#'   \item \code{sigma_scale}: A vector of scale parameters for the approximate inverse gamma posterior (sigma_tilde) of the continuous true latent values.
#'   \item \code{ELBO}: A vector of maximum ELBO at each generation.
#'   \item \code{niter}: The number of generations EVIL created.
#'   \item \code{interruption}: Whether the CHOMPER-EVIL is interrupted. The fitting is interrupted if the elapsed time reaches the maximum time limit.
#'   \item \code{maximum_elapsed_time}: The maximum elapsed time of a single CAVI iteration throughout the entire EVIL process.
#'   \item \code{elapsed_time}: The elapsed time of the entire EVIL process in seconds.
#' }
#'
#' @examples
#' # 1. Generate sample data for testing
#' sample_data <- generate_sample_data(
#'   n_entities = 10,
#'   n_files = 3,
#'   overlap_ratio = 0.7,
#'   discrete_columns = c(1, 2),
#'   discrete_levels = c(3, 3),
#'   continuous_columns = c(3, 4),
#'   continuous_params = matrix(c(0, 0, 1, 1), ncol = 2),
#'   distortion_ratio = c(0.1, 0.1, 0.1, 0.1)
#' )
#'
#' # 2. Get file information and remove `id` from the original data
#' n <- numeric(3)
#' x <- list()
#' for (i in 1:3) {
#'   n[i] <- nrow(sample_data[[i]])
#'   x[[i]] <- sample_data[[i]][, -1]
#' }
#' N <- sum(n)
#'
#' # 3. Set Hyperparameters
#' hyper_beta <- matrix(
#'   rep(c(N * 0.1 * 0.01, N * 0.1), 4),
#'   ncol = 2, byrow = TRUE
#' )
#'
#' hyper_sigma <- matrix(
#'   rep(c(0.01, 0.01), 2),
#'   ncol = 2, byrow = TRUE
#' )
#'
#' # 4. Fit CHOMPER-EVIL
#' result <- chomperEVIL(
#'   x = x,
#'   k = 3, # number of datasets
#'   n = n, # rows per dataset
#'   N = N, # columns per dataset
#'   p = 4, # fields per dataset
#'   M = c(3, 3), # categories for discrete fields
#'   discrete_fields = c(1, 2),
#'   continuous_fields = c(3, 4),
#'   hyper_beta = hyper_beta, # hyperparameter for distortion rate
#'   hyper_sigma = hyper_sigma, # hyperparameter for continuous fields
#'   hyper_phi = c(2.0, 2.0),
#'   hyper_tau = c(0.01, 0.01),
#'   hyper_epsilon_discrete = c(0, 0),
#'   hyper_epsilon_continuous = c(0.001, 0.001),
#'   n_threads = 1
#' )
#'
#' @export
chomperEVIL <- function(x, k, n, N, p, M, discrete_fields, continuous_fields,
                        hyper_beta, hyper_phi = c(), hyper_tau = c(),
                        hyper_epsilon_discrete = c(), hyper_epsilon_continuous = c(),
                        hyper_sigma = matrix(nrow = 0, ncol = 2),
                        overlap_prob = 0.5, n_parents = 5, n_children = 10,
                        tol_cavi = 1e-5, max_iter_cavi = 100, tol_evi = 1e-5,
                        max_iter_evi = 50, n_threads = 1,
                        max_time = 86400, custom_initializer = FALSE,
                        use_checkpoint = FALSE, initial_values = NULL,
                        checkpoint_values = NULL, verbose_internal = TRUE) {
  # Convert the field indexes to 0-based indexing
  discrete_fields <- discrete_fields - 1
  continuous_fields <- continuous_fields - 1

  if (any(discrete_fields == -1)) {
    n_discrete_fields <- 0
  } else {
    n_discrete_fields <- length(discrete_fields)
  }

  if (any(continuous_fields == -1)) {
    n_continuous_fields <- 0
  } else {
    n_continuous_fields <- length(continuous_fields)
  }

  if ((length(x) != k) || (length(n) != k)) {
    stop("Check the number of files")
  }

  n_cols <- numeric(k)
  n_rows <- numeric(k)
  for (i in 1:k) {
    n_cols[i] <- ncol(x[[i]])
    n_rows[i] <- nrow(x[[i]])
  }

  if (any(n_cols != p)) {
    stop("Check the number of columns in each file")
  }

  if (any(n_rows != n)) {
    stop("Check the number of rows in each file")
  }

  if (length(M) != n_discrete_fields) {
    stop("Check the number of discrete fields")
  }

  if (nrow(hyper_sigma) != n_continuous_fields) {
    stop("Check the number of continuous fields")
  }

  if ((n_discrete_fields + n_continuous_fields) != p) {
    stop("Check the number of fields")
  }

  if ((nrow(hyper_beta) != p) || (ncol(hyper_beta) != 2)) {
    stop("Check the dimension of hyperparameters")
  }

  if (length(hyper_phi) != n_discrete_fields) {
    warning(
      "hyperparameter (phi) is not provided properly. ",
      "The default value, (2.0, ..., 2.0), is used."
    )
    hyper_phi <- rep(2.0, n_discrete_fields)
  }

  if (length(hyper_tau) != n_discrete_fields) {
    warning(
      "hyperparameter (tau) is not provided properly. ",
      "The default value, (0.01, ..., 0.01), is used."
    )
    hyper_tau <- rep(0.01, n_discrete_fields)
  }

  if (length(hyper_epsilon_discrete) != n_discrete_fields) {
    warning(
      "hyperparameter (hitting range for discrete fields) is not provided properly. ",
      "The default value, (0, ..., 0) is used."
    )
    hyper_epsilon_discrete <- rep(0, n_discrete_fields)
  }

  if (length(hyper_epsilon_continuous) != n_continuous_fields) {
    warning(
      "hyperparameter (hitting range for continuous fields) is not provided properly. ",
      "The default value, (0.001, ..., 0.001), is used."
    )
    hyper_epsilon_continuous <- rep(0.001, n_continuous_fields)
  }

  temperature_parameter <- numeric(p)
  for (li in 1:n_discrete_fields) {
    temperature_parameter[discrete_fields[li] + 1] <- hyper_tau[li]
  }
  for (li in 1:n_continuous_fields) {
    temperature_parameter[continuous_fields[li] + 1] <- hyper_epsilon_continuous[li]
  }

  if (nrow(hyper_sigma) != n_continuous_fields) {
    warning(
      "hyperparameter (sigma) is not provided properly. ",
      "The default value, Inv-Gamma(0.01, 0.01) is used."
    )
    hyper_sigma <-
      matrix(
        rep(c(0.01, 0.01), n_continuous_fields),
        ncol = 2, byrow = TRUE
      )
  }

  if (custom_initializer && use_checkpoint) {
    stop("Checkpoint and Custom Initializer cannot be used at the same time.")
  }

  return(.EvolutionaryVI(
    x = x, k = k, n = n, N = N, p = p,
    discrete_fields = discrete_fields,
    n_discrete_fields = n_discrete_fields,
    M = M, continuous_fields = continuous_fields,
    n_continuous_fields = n_continuous_fields,
    hyper_beta = hyper_beta, hyper_sigma = hyper_sigma,
    hyper_phi = hyper_phi, hyper_tau = temperature_parameter,
    hyper_delta = hyper_epsilon_discrete, overlap_prob = overlap_prob,
    n_parents = n_parents, n_children = n_children,
    tol_cavi = tol_cavi, max_iter_cavi = max_iter_cavi,
    tol_evi = tol_evi, max_iter_evi = max_iter_evi, verbose = verbose_internal,
    n_threads = n_threads, max_time = max_time,
    custom_initializer = custom_initializer, use_checkpoint = use_checkpoint,
    initial_values = initial_values, checkpoint_values = checkpoint_values
  ))
}

#' @title CHOMPER with Markov chain Monte Carlo with Split and Merge Process
#' @description
#' Fit the CHOMPER model with Markov chain Monte Carlo (MCMC) with split and merge process to estimate the linkage structure across multiple datasets.
#' It returns the posterior samples of the linkage structure and other parameters of the CHOMPER model.
#'
#' @param x A list of data frames, each representing a dataset.
#' @param k The number of datasets to be linked.
#' @param n The number of rows in each dataset (vector of length k).
#' @param N The number of columns in each dataset.
#' @param p The number of fields in each dataset.
#' @param M The number of categories for each discrete field (vector of length of discrete fields).
#' @param discrete_fields The indexes of the discrete fields (1-based index).
#' @param continuous_fields The indexes of the continuous fields (1-based index).
#' @param hyper_beta The hyperparameters for the beta distribution (matrix of size p x 2).
#' @param hyper_phi The hyperparameters for softmax representation (vector of length of discrete fields).
#' @param hyper_tau The temperature parameter (vector of length of discrete fields).
#' @param hyper_epsilon_discrete The range parameter for the comprehensive hit of discrete fields (vector of length of discrete fields).
#' @param hyper_epsilon_continuous The range parameter for the comprehensive hit of continuous fields (vector of length of continuous fields).
#' @param hyper_sigma The hyperparameters for the Inverse Gamma distribution (matrix of size length of continuous fields x 2).
#' @param n_burnin The number of burn-in iterations for the MCMC.
#' @param n_gibbs The number of Gibbs sampling iterations for the MCMC.
#' @param n_split_merge The number of split and merge iterations for the MCMC.
#' @param max_time The maximum time limit for the execution in seconds.
#' @param custom_initializer Whether to use a custom initializer for the initial values.
#' @param use_checkpoint Whether to use a checkpoint.
#' @param initial_values The initial values for the parameters (optional).
#' @param checkpoint_values The checkpoint values for the parameters (optional).
#' @param verbose_internal Whether to print the internal C++ messages (TRUE: print, FALSE: not print).
#'
#' @return A list containing the posterior samples.
#' @return A list of the posterior samples and other information containing:
#' \itemize{
#'   \item \code{lambda}: A list of posterior samples (integer vectors) of the linkage structure.
#'   \item \code{z}: A list of posterior samples (binary matrices) of the distortion indicators.
#'   \item \code{y}: A list of posterior samples (matrices) of the true latent records.
#'   \item \code{beta}: A list of posterior samples (numeric vectors) of the distortion ratio.
#'   \item \code{theta}: A list of posterior samples (numeric vectors) of the probabilities of discrete true latent values.
#'   \item \code{eta}: A list of posterior samples (numeric vectors) of the mean of continuous true latent values.
#'   \item \code{sigma}: A list of posterior samples (numeric vectors) of the variance of continuous true latent values.
#'   \item \code{n_sample}: Total number of posterior samples after burn-in.
#'   \item \code{n_shift}: Total number of accepted split and merge results after burn-in.
#'   \item \code{elapsed_time}: The elapsed time of the entire MCMC process in seconds.
#'   \item \code{interruption}: Whether the CHOMPER-MCMC is interrupted. The fitting is interrupted if the elapsed time reaches the maximum time limit.
#' }
#'
#' @examples
#' # 1. Generate sample data for testing
#' sample_data <- generate_sample_data(
#'   n_entities = 10,
#'   n_files = 3,
#'   overlap_ratio = 0.7,
#'   discrete_columns = c(1, 2),
#'   discrete_levels = c(3, 3),
#'   continuous_columns = c(3, 4),
#'   continuous_params = matrix(c(0, 0, 1, 1), ncol = 2),
#'   distortion_ratio = c(0.1, 0.1, 0.1, 0.1)
#' )
#'
#' # 2. Get file information and remove `id` from the original data
#' n <- numeric(3)
#' x <- list()
#' for (i in 1:3) {
#'   n[i] <- nrow(sample_data[[i]])
#'   x[[i]] <- sample_data[[i]][, -1]
#' }
#' N <- sum(n)
#'
#' # 3. Set Hyperparameters
#' hyper_beta <- matrix(
#'   rep(c(N * 0.1 * 0.01, N * 0.1), 4),
#'   ncol = 2, byrow = TRUE
#' )
#'
#' hyper_sigma <- matrix(
#'   rep(c(0.01, 0.01), 2),
#'   ncol = 2, byrow = TRUE
#' )
#'
#' # 4. Fit CHOMPER-MCMC
#' result <- chomperMCMC(
#'   x = x,
#'   k = 3, # number of datasets
#'   n = n, # rows per dataset
#'   N = N, # columns per dataset
#'   p = 4, # fields per dataset
#'   M = c(3, 3), # categories for discrete fields
#'   discrete_fields = c(1, 2),
#'   continuous_fields = c(3, 4),
#'   hyper_beta = hyper_beta, # hyperparameter for distortion rate
#'   hyper_sigma = hyper_sigma, # hyperparameter for continuous fields
#'   hyper_phi = c(2.0, 2.0),
#'   hyper_tau = c(0.01, 0.01),
#'   hyper_epsilon_discrete = c(0, 0),
#'   hyper_epsilon_continuous = c(0.001, 0.001),
#'   n_burnin = 0,
#'   n_gibbs = 100,
#'   n_split_merge = 10
#' )
#'
#' @export
chomperMCMC <- function(x, k, n, N, p, M, discrete_fields, continuous_fields,
                        hyper_beta, hyper_phi = c(), hyper_tau = c(),
                        hyper_epsilon_discrete = c(), hyper_epsilon_continuous = c(),
                        hyper_sigma = matrix(nrow = 0, ncol = 2),
                        n_burnin = 1000, n_gibbs = 1000, n_split_merge = 10,
                        max_time = 86400, custom_initializer = FALSE,
                        use_checkpoint = FALSE, initial_values = NULL,
                        checkpoint_values = NULL, verbose_internal = TRUE) {
  # Convert the field indexes to 0-based indexing
  discrete_fields <- discrete_fields - 1
  continuous_fields <- continuous_fields - 1

  if (any(discrete_fields == -1)) {
    n_discrete_fields <- 0
  } else {
    n_discrete_fields <- length(discrete_fields)
  }

  if (any(continuous_fields == -1)) {
    n_continuous_fields <- 0
  } else {
    n_continuous_fields <- length(continuous_fields)
  }

  if ((length(x) != k) || (length(n) != k)) {
    stop("Check the number of files")
  }

  n_cols <- numeric(k)
  n_rows <- numeric(k)
  for (i in 1:k) {
    n_cols[i] <- ncol(x[[i]])
    n_rows[i] <- nrow(x[[i]])
  }

  if (any(n_cols != p)) {
    stop("Check the number of columns in each file")
  }

  if (any(n_rows != n)) {
    stop("Check the number of rows in each file")
  }

  if (length(M) != n_discrete_fields) {
    stop("Check the number of discrete fields")
  }

  if (nrow(hyper_sigma) != n_continuous_fields) {
    stop("Check the number of continuous fields")
  }

  if ((n_discrete_fields + n_continuous_fields) != p) {
    stop("Check the number of fields")
  }

  if ((nrow(hyper_beta) != p) || (ncol(hyper_beta) != 2)) {
    stop("Check the dimension of hyperparameters")
  }

  if (length(hyper_phi) != n_discrete_fields) {
    warning(
      "hyperparameter (phi) is not provided properly. ",
      "The default value, (2.0, ..., 2.0), is used."
    )
    hyper_phi <- rep(2.0, n_discrete_fields)
  }

  if (length(hyper_tau) != n_discrete_fields) {
    warning(
      "hyperparameter (tau) is not provided properly. ",
      "The default value, (0.01, ..., 0.01), is used."
    )
    hyper_tau <- rep(0.01, n_discrete_fields)
  }

  if (length(hyper_epsilon_discrete) != n_discrete_fields) {
    warning(
      "hyperparameter (hitting range for discrete fields) is not provided properly. ",
      "The default value, (0, ..., 0) is used."
    )
    hyper_epsilon_discrete <- rep(0, n_discrete_fields)
  }

  if (length(hyper_epsilon_continuous) != n_continuous_fields) {
    warning(
      "hyperparameter (hitting range for continuous fields) is not provided properly. ",
      "The default value, (0.001, ..., 0.001), is used."
    )
    hyper_epsilon_continuous <- rep(0.001, n_continuous_fields)
  }

  temperature_parameter <- numeric(p)
  for (li in 1:n_discrete_fields) {
    temperature_parameter[discrete_fields[li] + 1] <- hyper_tau[li]
  }
  for (li in 1:n_continuous_fields) {
    temperature_parameter[continuous_fields[li] + 1] <- hyper_epsilon_continuous[li]
  }

  if (nrow(hyper_sigma) != n_continuous_fields) {
    warning(
      "hyperparameter (sigma) is not provided properly. ",
      "The default value, Inv-Gamma(0.01, 0.01) is used."
    )
    hyper_sigma <-
      matrix(
        rep(c(0.01, 0.01), n_continuous_fields),
        ncol = 2, byrow = TRUE
      )
  }

  if (custom_initializer && use_checkpoint) {
    stop("Checkpoint and Custom Initializer cannot be used at the same time.")
  }

  return(.MCMC(
    x = x, k = k, n = n, N = N, p = p,
    discrete_fields = discrete_fields,
    n_discrete_fields = n_discrete_fields,
    M = M, continuous_fields = continuous_fields,
    n_continuous_fields = n_continuous_fields,
    hyper_beta = hyper_beta, hyper_sigma = hyper_sigma,
    hyper_phi = hyper_phi, hyper_tau = temperature_parameter,
    hyper_delta = hyper_epsilon_discrete, n_burnin = n_burnin,
    n_gibbs = n_gibbs, n_split_merge = n_split_merge,
    verbose = verbose_internal,
    max_time = max_time, custom_initializer = custom_initializer,
    use_checkpoint = use_checkpoint, initial_values = initial_values,
    checkpoint_values = checkpoint_values
  ))
}


#' @title CHOMPER with a single Coordinate Ascent Variational Inference
#' @description
#' Fit the CHOMPER model with a single Coordinate Ascent Variational Inference (CAVI) to estimate the linkage structure across multiple datasets.
#' It returns the approximate variational factors of the linkage structure that maximize the evidence lower bound (ELBO) and other parameters of the CHOMPER model.
#'
#' @param x A list of data frames, each representing a dataset.
#' @param k The number of datasets to be linked.
#' @param n The number of rows in each dataset (vector of length k).
#' @param N The number of columns in each dataset.
#' @param p The number of fields in each dataset.
#' @param M The number of categories for each discrete field (vector of length of discrete fields).
#' @param discrete_fields The indexes of the discrete fields (1-based index).
#' @param continuous_fields The indexes of the continuous fields (1-based index).
#' @param hyper_beta The hyperparameters for the beta distribution (matrix of size p x 2).
#' @param hyper_phi The hyperparameters for softmax representation (vector of length of discrete fields).
#' @param hyper_tau The temperature parameter (vector of length of discrete fields).
#' @param hyper_epsilon_discrete The range parameter for the comprehensive hit of discrete fields (vector of length of discrete fields).
#' @param hyper_epsilon_continuous The range parameter for the comprehensive hit of continuous fields (vector of length of continuous fields).
#' @param hyper_sigma The hyperparameters for the Inverse Gamma distribution (matrix of size length of continuous fields x 2).
#' @param overlap_prob The presumed probability of overlap across the datasets.
#' @param tol_cavi The tolerance for the coordinate ascent variational inference for the convergence.
#' @param max_iter_cavi The maximum number of iterations for the coordinate ascent variational inference.
#' @param max_time The maximum time limit for the execution in seconds.
#' @param custom_initializer Whether to use a custom initializer for the initial values.
#' @param use_checkpoint Whether to use a checkpoint.
#' @param initial_values The initial values for the parameters (optional).
#' @param checkpoint_values The checkpoint values for the parameters (optional).
#' @param verbose_internal Whether to print the internal C++ messages (TRUE: print, FALSE: not print).
#'
#' @return A list of the approximated parameters of the variational factors and other information containing:
#' \itemize{
#'   \item \code{nu}: A list of parameter matrices for the approximate multinomial posterior of the linkage structure.
#'   \item \code{omega}: A matrix of parameter vectors for the approximate beta posterior of the distortion ratio.
#'   \item \code{rho}: A list of parameter matrices for the approximate Bernoulli posterior of the distortion indicators.
#'   \item \code{gamma}: A list of parameter matrices for the approximate multinomial posterior of discrete true latent values.
#'   \item \code{alpha}: A list of parameter vectors for the approximate Dirichlet posterior (theta) of the discrete true latent values.
#'   \item \code{eta_tilde}: A matrix of parameter vectors for the mean of the approximate normal posterior of continuous true latent values.
#'   \item \code{eta_mean}: A vector of mean parameters for the approximate normal posterior (eta_tilde) of the continuous true latent values.
#'   \item \code{eta_var}: A vector of variance parameters for the approximate normal posterior (eta_tilde) of the continuous true latent values.
#'   \item \code{sigma_tilde}: A matrix of parameter vectors for the variance of the approximate normal posterior of continuous true latent values.
#'   \item \code{sigma_shape}: A vector of shape parameters for the approximate inverse gamma posterior (sigma_tilde) of the continuous true latent values.
#'   \item \code{sigma_scale}: A vector of scale parameters for the approximate inverse gamma posterior (sigma_tilde) of the continuous true latent values.
#'   \item \code{ELBO}: The final maximum ELBO from a single CAVI.
#'   \item \code{niter}: The number of iterations for a single CAVI.
#'   \item \code{interruption}: Whether the CHOMPER-CAVI is interrupted. The fitting is interrupted if the elapsed time reaches the maximum time limit.
#'   \item \code{cavi_elapsed_time}: The maximum elapsed time of a single CAVI iteration.
#'   \item \code{elapsed_time}: The elapsed time of the entire CAVI process in seconds.
#' }
#'
#' @examples
#' # 1. Generate sample data for testing
#' sample_data <- generate_sample_data(
#'   n_entities = 10,
#'   n_files = 3,
#'   overlap_ratio = 0.7,
#'   discrete_columns = c(1, 2),
#'   discrete_levels = c(3, 3),
#'   continuous_columns = c(3, 4),
#'   continuous_params = matrix(c(0, 0, 1, 1), ncol = 2),
#'   distortion_ratio = c(0.1, 0.1, 0.1, 0.1)
#' )
#'
#' # 2. Get file information and remove `id` from the original data
#' n <- numeric(3)
#' x <- list()
#' for (i in 1:3) {
#'   n[i] <- nrow(sample_data[[i]])
#'   x[[i]] <- sample_data[[i]][, -1]
#' }
#' N <- sum(n)
#'
#' # 3. Set Hyperparameters
#' hyper_beta <- matrix(
#'   rep(c(N * 0.1 * 0.01, N * 0.1), 4),
#'   ncol = 2, byrow = TRUE
#' )
#'
#' hyper_sigma <- matrix(
#'   rep(c(0.01, 0.01), 2),
#'   ncol = 2, byrow = TRUE
#' )
#'
#' # 4. Fit CHOMPER-CAVI
#' result <- chomperCAVI(
#'   x = x,
#'   k = 3, # number of datasets
#'   n = n, # rows per dataset
#'   N = N, # columns per dataset
#'   p = 4, # fields per dataset
#'   M = c(3, 3), # categories for discrete fields
#'   discrete_fields = c(1, 2),
#'   continuous_fields = c(3, 4),
#'   hyper_beta = hyper_beta, # hyperparameter for distortion rate
#'   hyper_sigma = hyper_sigma, # hyperparameter for continuous fields
#'   hyper_phi = c(2.0, 2.0),
#'   hyper_tau = c(0.01, 0.01),
#'   hyper_epsilon_discrete = c(0, 0),
#'   hyper_epsilon_continuous = c(0.001, 0.001),
#' )
#'
#' @export
chomperCAVI <- function(x, k, n, N, p, M, discrete_fields, continuous_fields,
                        hyper_beta, hyper_phi = c(), hyper_tau = c(),
                        hyper_epsilon_discrete = c(), hyper_epsilon_continuous = c(),
                        hyper_sigma = matrix(nrow = 0, ncol = 2),
                        overlap_prob = 0.5, tol_cavi = 1e-5, max_iter_cavi = 100,
                        max_time = 86400, custom_initializer = FALSE,
                        use_checkpoint = FALSE, initial_values = NULL,
                        checkpoint_values = NULL, verbose_internal = TRUE) {
  # Convert the field indexes to 0-based indexing
  discrete_fields <- discrete_fields - 1
  continuous_fields <- continuous_fields - 1

  if (any(discrete_fields == -1)) {
    n_discrete_fields <- 0
  } else {
    n_discrete_fields <- length(discrete_fields)
  }

  if (any(continuous_fields == -1)) {
    n_continuous_fields <- 0
  } else {
    n_continuous_fields <- length(continuous_fields)
  }

  if ((length(x) != k) || (length(n) != k)) {
    stop("Check the number of files")
  }

  n_cols <- numeric(k)
  n_rows <- numeric(k)
  for (i in 1:k) {
    n_cols[i] <- ncol(x[[i]])
    n_rows[i] <- nrow(x[[i]])
  }

  if (any(n_cols != p)) {
    stop("Check the number of columns in each file")
  }

  if (any(n_rows != n)) {
    stop("Check the number of rows in each file")
  }

  if (length(M) != n_discrete_fields) {
    stop("Check the number of discrete fields")
  }

  if (nrow(hyper_sigma) != n_continuous_fields) {
    stop("Check the number of continuous fields")
  }

  if ((n_discrete_fields + n_continuous_fields) != p) {
    stop("Check the number of fields")
  }

  if ((nrow(hyper_beta) != p) || (ncol(hyper_beta) != 2)) {
    stop("Check the dimension of hyperparameters")
  }

  if (length(hyper_phi) != n_discrete_fields) {
    warning(
      "hyperparameter (phi) is not provided properly. ",
      "The default value, (2.0, ..., 2.0), is used."
    )
    hyper_phi <- rep(2.0, n_discrete_fields)
  }

  if (length(hyper_tau) != n_discrete_fields) {
    warning(
      "hyperparameter (tau) is not provided properly. ",
      "The default value, (0.01, ..., 0.01), is used."
    )
    hyper_tau <- rep(0.01, n_discrete_fields)
  }

  if (length(hyper_epsilon_discrete) != n_discrete_fields) {
    warning(
      "hyperparameter (hitting range for discrete fields) is not provided properly. ",
      "The default value, (0, ..., 0) is used."
    )
    hyper_epsilon_discrete <- rep(0, n_discrete_fields)
  }

  if (length(hyper_epsilon_continuous) != n_continuous_fields) {
    warning(
      "hyperparameter (hitting range for continuous fields) is not provided properly. ",
      "The default value, (0.001, ..., 0.001), is used."
    )
    hyper_epsilon_continuous <- rep(0.001, n_continuous_fields)
  }

  temperature_parameter <- numeric(p)
  for (li in 1:n_discrete_fields) {
    temperature_parameter[discrete_fields[li] + 1] <- hyper_tau[li]
  }
  for (li in 1:n_continuous_fields) {
    temperature_parameter[continuous_fields[li] + 1] <- hyper_epsilon_continuous[li]
  }

  if (nrow(hyper_sigma) != n_continuous_fields) {
    warning(
      "hyperparameter (sigma) is not provided properly. ",
      "The default value, Inv-Gamma(0.01, 0.01) is used."
    )
    hyper_sigma <-
      matrix(
        rep(c(0.01, 0.01), n_continuous_fields),
        ncol = 2, byrow = TRUE
      )
  }

  if (custom_initializer && use_checkpoint) {
    stop("Checkpoint and Custom Initializer cannot be used at the same time.")
  }

  return(.CoordinateAscentVI(
    x = x, k = k, n = n, N = N, p = p,
    discrete_fields = discrete_fields,
    n_discrete_fields = n_discrete_fields,
    M = M, continuous_fields = continuous_fields,
    n_continuous_fields = n_continuous_fields,
    hyper_beta = hyper_beta, hyper_sigma = hyper_sigma,
    hyper_phi = hyper_phi, hyper_tau = temperature_parameter,
    hyper_delta = hyper_epsilon_discrete, overlap_prob = overlap_prob,
    tol_cavi = tol_cavi, max_iter_cavi = max_iter_cavi,
    verbose = verbose_internal, max_time = max_time,
    custom_initializer = custom_initializer, use_checkpoint = use_checkpoint,
    initial_values = initial_values, checkpoint_values = checkpoint_values
  ))
}
