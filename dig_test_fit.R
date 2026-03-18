# CHOMPER-DIG Test Code
library(chomper)

args <- commandArgs(TRUE)

sdx <- as.integer(args[1])
approach <- as.character(args[2])

sdx <- 1
approach <- "Conservative"
i_sim <- 1
model <- "DIG"
overlap_ratio <- "high"

for (i_sim in 1:30) {
    # Load data
    # We offer 3 different files depending on the overlap ratio.
    # Each file is a list fo 30 sets of synthetic data.
    #   high (70%) overlap ratio: simulation-high-overlap.rds
    #   medium (50%) overlap ratio: simulation-medium-overlap.rds
    #   low (30%) overlap ratio: simulation-low-overlap.rds
    data_all <- readRDS(paste0(
        "~/dev/chomper.reproduce/data/simulation-",
        overlap_ratio, "-overlap.rds"
    ))

    dat <- data_all[[i_sim]]

    k <- length(dat) # number of files
    n <- numeric(k) # number of records in each file
    for (i in 1:k) {
        n[i] <- nrow(dat[[i]])
    }
    N <- sum(n) # total number of records
    N_max <- N # total number of records, which is used for setting a prior

    # Select variables for the estimation
    #   X1 ~ X3: multinomial, single truth
    #   X4 ~ X5: multinomial, multiple truths
    #   X6 ~ X7: Gaussian, multiple truths
    if (sdx == 1) {
        if (approach == "Conservative") {
            common_fields <- c("X1", "X2", "X3")
            # Set the types of variables.
            # If one type of missing, please set it with 0
            discrete_fields <- 1:3
            n_discrete_fields <- 3
        } else {
            common_fields <- c("X1", "X2", "X3", "X4", "X5")
            discrete_fields <- 1:5
            n_discrete_fields <- 5
        }
        continuous_fields <- 0 # If no continuous fields are used, replace them with 0,
        n_continuous_fields <- 0 # and replace this argument with 0

        if (approach %in% c("Conservative", "Naive")) {
            hyper_epsilon_discrete <-
                rep(0, n_discrete_fields) # hitting range for categorical fields
            # As no categorical variable has multiple truths,
            # we set epsilon as 0.
        } else if (approach == "Oracle") {
            hyper_epsilon_discrete <-
                c(0, 0, 0, 1, 1)
        } else if (approach == "Comprehensive") {
            hyper_epsilon_discrete <-
                c(0, 0, 0, 2, 2)
        } else {
            stop("Choose an appropriate approach.")
        }

        hyper_epsilon_continuous <-
            rep(0.01, n_continuous_fields) # hitting range for continuous fields.
    } else if (sdx == 2) {
        common_fields <- c("X1", "X2", "X3", "X6", "X7")

        # Set the types of variables.
        # If one type of missing, please set it with 0
        discrete_fields <- 1:3
        n_discrete_fields <- 3
        continuous_fields <- 4:5 # If no continuous fields are used, replace them with 0,
        n_continuous_fields <- 2 # and replace this argument with 0

        hyper_epsilon_discrete <-
            rep(0, n_discrete_fields) # hitting range for categorical fields
        # As no categorical variable has multiple truths,
        # we set epsilon as 0.

        if (approach == "Naive") {
            varepsilon_ell <- 0.001
        } else if (approach == "Oracle") {
            varepsilon_ell <- 0.1
        } else if (approach == "Comprehensive") {
            varepsilon_ell <- 0.5
        } else if (approach == "Vague") {
            varepsilon_ell <- 1.0
        } else {
            stop("Choose an appropriate approach.")
        }
        hyper_epsilon_continuous <-
            rep(varepsilon_ell, n_continuous_fields) # hitting range for continuous fields.
    } else {
        stop("Choose an appropriate scenario.")
    }

    p <- length(common_fields) # number of variables

    x <- list()
    for (i in 1:k) {
        x[[i]] <- as.matrix(dat[[i]][, common_fields])
    }

    M <- rep(8, n_discrete_fields) # number of levels in categorical variables
    hyper_beta <- matrix(
        rep(c(N_max * 0.1 * 0.01, N_max * 0.1), p),
        ncol = 2, byrow = TRUE
    ) # prior distribution for distortion rate, following \cite{marchant2021}
    hyper_sigma <- matrix(
        rep(c(0.01, 0.01), n_continuous_fields),
        ncol = 2, byrow = TRUE
    ) # prior distribution for the variance of Gaussian, setting non-informative
    hyper_phi <- rep(2.0, n_discrete_fields) # hyperparameter for categorical fields
    hyper_tau <- rep(0.01, n_discrete_fields) # hyperparameter for categorical fields

    # Set seed
    set.seed(12345)

    # Fit the model
    res <- chomperDIG(
        x = x, k = k, n = n, N = N, p = p, M = M,
        discrete_fields = discrete_fields,
        continuous_fields = continuous_fields,
        hyper_beta = hyper_beta,
        hyper_phi = hyper_phi,
        hyper_tau = hyper_tau,
        hyper_epsilon_discrete = hyper_epsilon_discrete,
        hyper_epsilon_continuous = hyper_epsilon_continuous,
        hyper_sigma = hyper_sigma,
        decaying_upper_bound = 100.0,
        n_burnin = 0, # number of burn-in samples
        n_gibbs = 10000, # number of MCMC samples to store
        batch_size = 50,
        n_epochs = 5,
        n_split_merge = 1000,
        max_time = 86400 # maximum time in second for MCMC
    )

    # performance samples:
    # 9001 - 10000
    # 29001 - 30000
    # 49001 - 50000

    fname <- paste0(
        model, "-", overlap_ratio,
        "-SCE", sdx, "-", approach,
        "-SIM", sprintf("%02d", i_sim)
    )

    # Save the result
    # This data will be used for the estimation and evaluating the performance.
    saveRDS(
        res,
        paste0(
            "/Volumes/Extreme Pro/research/dig.test/",
            fname, ".rds"
        )
    )
}
