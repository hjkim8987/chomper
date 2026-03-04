#' @title Calculate the posterior similarity matrix
#' @description
#' This function returns a posterior similarity matrix based on the parameters of the approximated variational factor, nu, obtained from either \code{chomperEVIL} or \code{chomperCAVI}.
#'
#' @param probs_field a list of matrices with posterior probabilities
#'
#' @return a posterior similarity matrix of all possible pairs
#'
#' @examples
#' # 1. Create an approximate posterior distribution of linkage structure
#' n_file1 <- 2
#' n_file2 <- 3
#'
#' nu1 <- matrix(runif(n_file1^2 * n_file2), nrow = n_file1)
#' for (i in 1:n_file1) {
#'   nu1[i, ] <- nu1[i, ] / sum(nu1[i, ])
#' }
#'
#' nu2 <- matrix(runif(n_file1 * n_file2^2), nrow = n_file2)
#' for (i in 1:n_file2) {
#'   nu2[i, ] <- nu2[i, ] / sum(nu2[i, ])
#' }
#'
#' # 2. Convert into the appropriate type to run a function
#' approximate_posterior <- list(nu1, nu2)
#'
#' # 3. Calculate a posterior similarity matrix
#' psm_vi(approximate_posterior)
#'
#' @export
psm_vi <- function(probs_field) {
  result <- .psm_vi(probs_field)
  n <- nrow(result)
  return(round(result, nchar(n) + 1))
}

#' @title Calculate the posterior similarity matrix
#' @description
#' This function returns a posterior similarity matrix based on the MCMC samples of lambda, obtained from \code{chomperMCMC}.
#'
#' @param samples a total number of records by number of MCMC samples matrix with MCMC samples
#'
#' @return a posterior similarity matrix of all possible pairs
#'
#' @examples
#' # 1. Create a matrix with posterior samples of linkage structure
#' n_file1 <- 2
#' n_file2 <- 3
#'
#' number_of_records <- n_file1 + n_file2
#'
#' number_of_samples <- 10
#' lambda_matrix <- matrix(nrow = number_of_samples, ncol = number_of_records)
#' for (i in 1:number_of_samples) {
#'   lambda_matrix[i, ] <- sample(1:number_of_records, number_of_records, TRUE)
#' }
#'
#' # 2. Calculate a posterior similarity matrix
#' psm_mcmc(lambda_matrix)
#'
#' @export
psm_mcmc <- function(samples) {
  result <- .psm_mcmc(samples)
  n <- nrow(result)
  return(round(result, nchar(n) + 1))
}

#' @title Flatten the posterior samples, lambda, into a matrix
#' @description
#' This function converts a list of posterior samples of lambda into a matrix.
#' Before calculating the posterior similarity matrix using \code{psm_mcmc}, it is necessary to flatten the posterior samples into a matrix.
#'
#' @param samples a list of MCMC samples
#' @param k number of files to be linked
#' @param N total number of records
#'
#' @return an N by number of MCMC samples matrix
#'
#' @examples
#' # 1. Create a list of posterior samples of linkage structure
#' number_of_files <- 2
#'
#' n_file1 <- 2
#' n_file2 <- 3
#'
#' number_of_records <- n_file1 + n_file2
#'
#' number_of_samples <- 10
#' lambda <- list()
#' for (i in 1:number_of_samples) {
#'   lambda[[i]] <- list(
#'     sample(1:number_of_records, n_file1, TRUE),
#'     sample(1:number_of_records, n_file2, TRUE)
#'   )
#' }
#'
#' # 2. Converts a list of posterior samples of lambda into a matrix
#' flatten_posterior_samples(lambda, number_of_files, number_of_records)
#'
#' @export
flatten_posterior_samples <- function(
    samples, k, N) {
  return(.flatten_posterior_samples(samples, k, N))
}

#' @title Evaluate the performance of the linkage structure estimation
#' @description
#' Based on the true linkage structure and the estimate, it will calculate several metrics including true positive, true negative, false positive, false negative, false positive rate, and false negative rate.
#' This package recommends using the output of \code{links} function of the \code{blink} package as an argument to \code{performance} function.
#'
#' @param estimation estimated linkage structure
#' @param truth true linkage structure
#' @param N total number of records
#' @param return_matrix if true, it also returns the matrix of linkage structure
#'
#' @return a list with performance metrics. If \code{return_matrix} is true, it also returns the matrix of linkage structure used for the evaluation.
#'
#' @examples
#' # 1. True linkage structure
#' total_number_of_records <- 6
#'
#' truth <- matrix(
#'   list(c(1), c(2, 4), c(3), c(2, 4), c(5, 6), c(5, 6))
#' )
#'
#' # 2. Estimated linkage structure
#' estimation <- matrix(
#'   list(c(1), c(2, 4), c(3), c(2, 4), c(5), c(6))
#' )
#'
#' # 3. Calculate performance metrics
#' performance(estimation, truth, total_number_of_records, FALSE)
#'
#' @export
performance <- function(estimation, truth, N, return_matrix = FALSE) {
  return(.evaluate_performance(estimation, truth, N, return_matrix))
}
