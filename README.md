
# chomper

<!-- badges: start -->
[![CRAN](https://www.r-pkg.org/badges/version/chomper?color=orange)](https://cran.r-project.org/package=chomper)
[![download](http://cranlogs.r-pkg.org/badges/grand-total/chomper?color=blue)](https://cran.r-project.org/package=chomper)
[![R-CMD-check](https://github.com/hjkim8987/chomper/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/hjkim8987/chomper/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

## Overview

`chomper` is an R package that provides a Comprehensive Hit Or Miss Entity Resolution (CHOMPER) models.

## Key Features

- **Multiple Inference Approaches**: Implements three inference methods:
  - **MCMC**: Markov Chain Monte Carlo with split and merge process
  - **EVIL**: Evolutionary Variational Inference for record Linkage
  - **CAVI**: Single Coordinate Ascent Variational Inference  

- **Locally Varying Hit Mechanism**: Accounts for the attributes with multiple truths
- **Flexible Data Support**: Handles both discrete and continuous fields
- **Parallel Computing**: Multi-threading support for faster EVIL estimation

## Installation

You can install `chomper` from [GitHub](https://github.com/hjkim8987/chomper) with:

```r
# install.packages("devtools")
devtools::install_github("hjkim8987/chomper", dependencies = TRUE, build_vignettes = TRUE)
```

## Quick Start

```r
library(chomper)

# Generate sample data for testing
sample_data <- generate_sample_data(
  n_entities = 100,
  n_files = 3,
  overlap_ratio = 0.7,
  discrete_columns = c(1, 2),
  discrete_levels = c(5, 5),
  continuous_columns = c(3, 4),
  continuous_params = matrix(c(0, 0, 1, 1), ncol = 2),
  distortion_ratio = c(0.1, 0.1, 0.1, 0.1)
)

# Get file information and drop "id" column
n <- numeric(3)
x <- list()
for (i in 1:3) {
  n[i] <- nrow(sample_data[[i]])
  x[[i]] <- sample_data[[i]][, colnames(sample_data[[i]]) != "id", drop = FALSE]
}
N <- sum(n)

# Set Hyperparameters
hyper_beta <- matrix(
  rep(c(N * 0.1 * 0.01, N * 0.1), 4),
  ncol = 2, byrow = TRUE
)

hyper_sigma <- matrix(
  rep(c(0.01, 0.01), 2),
  ncol = 2, byrow = TRUE
)

# Perform record linkage using EVIL
result <- chomperEVIL(
  x = x,
  k = 3,  # number of datasets
  n = n,  # rows per dataset
  N = N,  # columns per dataset
  p = 4,  # fields per dataset
  M = c(5, 5),  # categories for discrete fields
  discrete_fields = c(1, 2),
  continuous_fields = c(3, 4),
  hyper_beta = hyper_beta,   # hyperparameter for distortion rate
  hyper_sigma = hyper_sigma, # hyperparameter for continuous fields
  n_threads = 4
)

# Performance evaluation
psm_ <- psm_vi(result$nu) # Calculate a posterior similarity matrix

# install.pakcages("salso")
library(salso)

salso_estimate <- salso(psm_,
  loss = binder(),
  maxZealousAttempts = 0, probSequentialAllocation = 1
) # Find a Bayes estimate that minimizes Binder's loss

linkage_structure <- list()
for (ll in seq_along(salso_estimate)) {
  linkage_structure[[ll]] <- which(salso_estimate == salso_estimate[ll])
}
linkage_estimation <- matrix(linkage_structure)

# install.packages("blink")
library(blink)

key_temp <- c()
for (i in 1:3) {
  key_temp <- c(key_temp, sample_data[[i]][, "id"])
}

truth_binded <- matrix(key_temp, nrow = 1)
linkage_structure_true <- links(truth_binded, TRUE, TRUE)
linkage_truth <- matrix(linkage_structure_true)

perf <- performance(linkage_estimation, linkage_truth, N)
print(perf)
```

## Main Functions

### Core Inference Functions

- `chomperMCMC()`: Markov Chain Monte Carlo
- `chomperEVIL()`: Evolutionary Variational Inference for record Linkage
- `chomperCAVI()`: Coordinate Ascent Variational Inference

### Data Generation and Utilities

- `generate_sample_data()`: Create synthetic data for testing and validation
- `flatten_posterior_samples()`: Flatten posterior samples for obtaining a posterior similarity matrix

### Evaluation and Performance

- `psm_mcmc()`: Posterior similarity matrix for MCMC results
- `psm_vi()`: Posterior similarity matrix for variational inference
- `performance()`: Evaluate performance of estimation

## License

This package is licensed under the GNU General Public License v3.0 - see the [LICENSE](https://github.com/hjkim8987/chomper/blob/main/LICENSE.md) file for details.

## Authors

- **Hyungjoon Kim** - *Maintainer* - [GitHub](https://github.com/hjkim8987)
- **Andee Kaplan** - *Contributor* - [GitHub](https://github.com/andeek)
- **Matthew Koslovsky** - *Contributor* - [GitHub](https://github.com/mkoslovsky)
