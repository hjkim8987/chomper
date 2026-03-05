#' @title Generate synthetic data for record linkage
#' @description
#' Generate synthetic data for record linkage with given number of entities, files, and overlap ratio.
#' Each variable can follow either a multinomial or a Gaussian distribution.
#' User can specify the existence of multiple truths for each variable.
#'
#' @param n_entities The number of entities.
#' @param n_files The number of files.
#' @param overlap_ratio The ratio of overlapping entities across the files.
#' @param discrete_columns The indices of the discrete columns (1-based index).
#' @param discrete_levels The levels of the discrete columns (vector of length of discrete columns).
#' @param continuous_columns The indices of the continuous columns (1-based index).
#' @param continuous_params The parameters of the continuous columns (matrix of size length of continuous columns x 2).
#' @param distortion_ratio The distortion ratio of the columns (vector of length of total columns).
#' @param discrete_fuzziness The configuration of the multiple truths of the discrete columns (optional).
#' @param continuous_fuzziness The configuration of the multiple truths of the continuous columns (optional).
#'
#' @return A list of matrices containing the noisy synthetic data. Each matrix represents a file.
#'
#' @examples
#' # 1. Set number of entities, files, and overlap ratio
#' n_entities <- 25
#' n_files <- 2
#' overlap_ratio <- 0.9
#'
#' # 2. Set attributes information
#' discrete_columns <- 1:4
#' discrete_levels <- rep(5, 4)
#' continuous_columns <- 5:6
#' continuous_params <-
#'   matrix(c(0, 10, 10, 10),
#'     ncol = 2, byrow = TRUE # means and variances
#'   )
#'
#' # 3. Set distortion ratio and fuzziness information
#' distortion_ratio <- rep(0.01, 6)
#' discrete_fuzziness <- matrix(c(4, 1),
#'   ncol = 2, byrow = TRUE
#' )
#' continuous_fuzziness <- matrix(c(5, 0.5^2, 6, 0.5^2),
#'   ncol = 2, byrow = TRUE
#' )
#'
#' # 4. Generate synthetic data
#' simulation_data <- generate_sample_data(
#'   n_entities, n_files, overlap_ratio,
#'   discrete_columns, discrete_levels,
#'   continuous_columns, continuous_params,
#'   distortion_ratio,
#'   discrete_fuzziness, continuous_fuzziness
#' )
#'
#' @export
generate_sample_data <- function(
    n_entities, n_files, overlap_ratio,
    discrete_columns, discrete_levels, continuous_columns, continuous_params,
    distortion_ratio, discrete_fuzziness = NULL, continuous_fuzziness = NULL) {
  add_discrete_fuzziness <- function(val, delta, max_val) {
    from_ <- val - delta
    to_ <- val + delta

    if (from_ < 1) {
      from_ <- 1
    }
    if (to_ > max_val) {
      to_ <- max_val
    }

    res <- sample(seq(from = from_, to = to_, by = 1), 1)

    return(res)
  }

  id_entity <- 1:n_entities
  id_file <- 1:n_files

  n_discrete <- length(discrete_columns)
  n_continuous <- length(continuous_columns)

  if (n_discrete != length(discrete_levels)) {
    stop(
      paste0(
        "Number of discrete columns (", n_discrete,
        ") does not match the number of levels per variable (",
        length(discrete_levels), ")."
      )
    )
  }

  if (n_continuous != nrow(continuous_params)) {
    stop(
      paste0(
        "Number of continuous columns (", n_continuous,
        ") does not match the number of parameters (",
        nrow(continuous_params), ")."
      )
    )
  }

  n_fields <- n_discrete + n_continuous
  if (n_fields != length(distortion_ratio)) {
    stop(
      paste0(
        "Number of fields (", n_fields,
        ") does not match the number of distortion ratios (",
        length(distortion_ratio), ")."
      )
    )
  }
  columns_names <- paste0("X", sprintf("%02d", 1:n_fields))

  levels_per_field <- list()
  for (l in seq_along(discrete_levels)) {
    levels_per_field[[l]] <- discrete_levels[[l]]
  }

  Y <- matrix(0, nrow = n_entities, ncol = n_fields)
  for (i in 1:n_entities) {
    for (ldx in seq_along(discrete_columns)) {
      l <- discrete_columns[ldx]
      Y[i, l] <- sample(levels_per_field[[ldx]], 1)
    }

    for (ldx in seq_along(continuous_columns)) {
      l <- continuous_columns[ldx]
      Y[i, l] <-
        stats::rnorm(1, continuous_params[ldx, 1], sqrt(continuous_params[ldx, 2]))
    }
  }

  id_overlap <-
    sample(id_entity, as.integer(n_entities * overlap_ratio), replace = FALSE)
  X_original <- matrix(nrow = 0, ncol = n_fields)
  id_expanded <- c()
  file_index <- c()
  for (j in id_entity) {
    if (j %in% id_overlap) {
      if (n_files == 2) {
        k <- 2
      } else {
        k <- sample(id_file[2:n_files], 1)
      }
      fdx <- sort(sample(id_file, k))

      id_expanded <- c(id_expanded, rep(j, k))
      X_original <- rbind(
        X_original, matrix(rep(Y[j, ], k), ncol = n_fields, byrow = TRUE)
      )
      file_index <- c(file_index, fdx)
    } else {
      id_expanded <- c(id_expanded, j)
      X_original <- rbind(X_original, Y[j, ])
      file_index <- c(file_index, sample(id_file, 1))
    }
  }

  n_records <- nrow(X_original)

  fuzzy_columns <- c()
  if (!is.null(discrete_fuzziness)) {
    fuzzy_columns <- c(fuzzy_columns, discrete_fuzziness[, 1])

    for (j in 1:n_records) {
      for (ldx in 1:nrow(discrete_fuzziness)) {
        l <- discrete_fuzziness[ldx, 1]
        delta <- discrete_fuzziness[ldx, 2]

        X_original[j, l] <-
          add_discrete_fuzziness(X_original[j, l], delta, discrete_levels[l])
      }
    }
  }

  if (!is.null(continuous_fuzziness)) {
    fuzzy_columns <- c(fuzzy_columns, continuous_fuzziness[, 1])

    for (j in 1:n_records) {
      for (ldx in 1:nrow(continuous_fuzziness)) {
        l <- continuous_fuzziness[ldx, 1]
        delta <- continuous_fuzziness[ldx, 2]

        X_original[j, l] <-
          stats::rnorm(1, X_original[j, l], sqrt(delta))
      }
    }
  }

  X_noise <- X_original
  for (j in 1:n_records) {
    for (ldx in seq_along(discrete_columns)) {
      l <- discrete_columns[ldx]

      if (l %in% fuzzy_columns) {
        next
      }

      if (stats::runif(1) < distortion_ratio[l]) {
        X_noise[j, l] <- sample(levels_per_field[[ldx]], 1)
      }
    }

    for (ldx in seq_along(continuous_columns)) {
      l <- continuous_columns[ldx]

      if (l %in% fuzzy_columns) {
        next
      }

      if (stats::runif(1) < distortion_ratio[l]) {
        X_noise[j, l] <-
          stats::rnorm(1, continuous_params[ldx, 1], sqrt(continuous_params[ldx, 2]))
      }
    }
  }

  X_original <- cbind(id_expanded, file_index, X_original)
  colnames(X_original) <- c("id", "file", columns_names)
  colnames(X_noise) <- columns_names

  df_original <- list()
  df_noise <- list()
  df_return <- list()
  for (i in 1:n_files) {
    loc <- which(X_original[, 2] == i)

    df_original[[i]] <- X_original[loc, ]
    df_noise[[i]] <- X_noise[loc, ]
    df_return[[i]] <- cbind(X_original[loc, "id"], X_noise[loc, ])
    colnames(df_return[[i]]) <- c("id", columns_names)
  }

  return(df_return)
}
