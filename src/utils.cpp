#include <RcppArmadillo.h>

#include "utils.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

/******************************************************************************
    Snippet:
        A collection of commonly used functions,
        including matrix manipulation and operation.
******************************************************************************/
// Create a matrix of possible pairs from a vector [0, ..., n - 1]
// That is, construct a matrix of all possible n choose 2
//
// @param n int size of a vector
// @param start starting index
//        0: [0, ..., n - 1], 1: [1, ..., n]
// @return matrix of pairs
arma::mat create_combination(int n, int start) {
  arma::mat result(n * (n - 1) / 2, 2);
  int row = 0;
  for (int i = start; i < n; i++) {
    if (start == 0) {
      for (int j = i + 1; j < n; j++) {
        result(row, 0) = i;
        result(row++, 1) = j;
      }
    } else {
      for (int j = i + 1; j <= n; j++) {
        result(row, 0) = i;
        result(row++, 1) = j;
      }
    }
  }
  return result;
}

// Construct a matrix by appending matrices in a field
//
// @param x a field of matrices
// @return matrix version of data
arma::mat compile_matrix(arma::field<arma::mat> x) {
  int k = x.n_elem;
  int N = 0;
  int p = x(0).n_cols;

  for (int i = 0; i < k; i++) {
    N += x(i).n_rows;
  }

  arma::mat result(N, p);

  int n_i = 0;
  for (int i = 0; i < k; i++) {
    result.rows(n_i, n_i + x(i).n_rows - 1) = x(i);
    n_i += x(i).n_rows;
  }

  return result;
}

// Construct a vector by appending vectors in a field
//
// @param x a field of vectors
// @return vector version of data
arma::vec compile_vector(arma::field<arma::vec> x) {
  int k = x.n_elem;

  int N = 0;
  for (int i = 0; i < k; i++) {
    N += x(i).n_elem;
  }

  arma::vec result(N, arma::fill::zeros);
  int current_position = 0;
  for (int i = 0; i < k; i++) {
    const arma::vec& x_i = x(i);
    int n_i = x_i.n_elem;

    if (n_i > 0) {
      result.subvec(current_position, current_position + n_i - 1) = x_i;
      current_position += n_i;
    }
  }

  return result;
}

// Split a matrix into k matrices by row
//
// @param x a matrix
// @param n vector of number of rows for each matrix
// @param k length of a vector n (number of matrices to be splitted)
// @return splited_matrix a field of matrices
arma::field<arma::mat> split_matrix(arma::mat x, arma::vec n, int k) {
  arma::field<arma::mat> result(k);

  int start_row = 0;
  for (int i = 0; i < k; i++) {
    int end_row = start_row + n(i) - 1;

    result(i) = x.rows(start_row, end_row);

    start_row = end_row + 1;
  }

  return result;
}

// Create a matrix with records indexes
// Each row has its corresponding record index i and j
//
// @param k number of files
// @param n vector of number of records in each file
// @return a sum(n) by 2 index matrix
arma::mat create_index_matrix(int k, arma::vec n) {
  arma::mat result(sum(n), 2);
  int jprime = 0;
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n(i); j++) {
      result(jprime, 0) = i;
      result(jprime, 1) = j;
      jprime += 1;
    }
  }
  return result;
}

// Generate a random order vector from [0, ..., N - 1]
//
// @param N length of a vector
// @param size size of the generated random vector
// @return a size by 1 integer vector
arma::uvec sample_index(int N, int size) {
  IntegerVector index = sample(N, size) - 1;
  return as<arma::uvec>(wrap(index));
}

// Sample a integer from an Armadillo vector
//
// @param x an Armadillo vector
// @return a random integer from the vector
int sample_integer(arma::uvec x) {
  arma::uvec samples = sample_index(x.n_elem, 1);
  return x(samples(0));
}

// Sample rows from an index matrix based on sampling weights
//
// @param index_matrix an index matrix
// @param sampling_weights a vector of sampling weights
// @param batch_size size of the sampled matrix
// @param N total number of records
// @return a sampled matrix
arma::mat sample_index_matrix(arma::mat index_matrix,
                              arma::vec sampling_weights, int batch_size,
                              int N) {
  IntegerVector indices = seq(0, N - 1);

  IntegerVector sampled_indices =
      sample(indices, batch_size, false,
             NumericVector(sampling_weights.begin(), sampling_weights.end()));
  arma::uvec arma_indices = as<arma::uvec>(sampled_indices);

  arma::mat sampled_matrix = index_matrix.rows(arma_indices);
  return sampled_matrix;
}

// Find the indexes of the maximum values in each row of matrices of a field
//
// @param probs a field of matrices
// @param k number of matrices
// @param n vector of number of rows in each matrix
// @return a field of vectors with the indexes of the maximum values
arma::field<arma::uvec> find_max_indexes(arma::field<arma::mat> probs, int k,
                                         arma::vec n) {
  arma::field<arma::uvec> result(k);
  for (int i = 0; i < k; i++) {
    arma::uvec indexes(n[i]);
    for (int j = 0; j < n[i]; j++) {
      NumericVector probs_ij = as<NumericVector>(wrap(probs[i].row(j)));
      indexes[j] = which_max(probs_ij);
    }
    result[i] = indexes;
  }
  return result;
}

// Find the top "n" indexes from two vectors
//
// @param x a vector
// @param y a vector
// @param n number of top indexes to find
// @return a vector of the top "n" indexes
IntegerVector get_top_n_indexes(NumericVector x, NumericVector y, int n) {
  // Get the sizes of the two vectors
  int n_x = x.size();
  int n_y = y.size();
  int n_total = n_x + n_y;

  // Combine the two vectors into a single vector
  NumericVector z(n_total);
  for (int i = 0; i < n_x; i++) {
    z[i] = x[i];
  }
  for (int i = 0; i < n_y; i++) {
    z[n_x + i] = y[i];
  }

  // Initialize the result vector and the visited vector,
  // and find the top "n" indexes
  IntegerVector result(n);
  std::vector<bool> visited(n_total, false);

  for (int i = 0; i < n; i++) {
    // Find the maximum value and index in the combined vector
    double max_val = R_NegInf;
    int max_idx = -1;

    for (int j = 0; j < n_total; j++) {
      if (!visited[j]) {
        if ((max_idx == -1) || (z[j] > max_val)) {
          max_val = z[j];
          max_idx = j;
        }
      }
    }

    if (max_idx != -1) {
      result[i] = max_idx;
      visited[max_idx] = true;
    }
  }

  return result;
}

// Swap the values of two elements in a row vector
//
// @param x a row vector
// @param i1 index of the first element
// @param i2 index of the second element
// @return a row vector with the values of the two elements swapped
arma::rowvec swap_row_values(arma::rowvec x, int i1, int i2) {
  arma::rowvec result = x;

  double aux = result[i1];
  result[i1] = x[i2];
  result[i2] = aux;

  return result;
}

// Create a matrix with the given number of rows and columns
//
// @param n_rows number of rows
// @param n_cols number of columns
// @return a matrix with the given number of rows and columns
arma::mat create_matrix(int n_rows, int n_cols) {
  arma::mat result(n_rows, n_cols);
  return result;
}

// Create a vector with the given length
//
// @param n_length length of the vector
// @return a vector with the given length
arma::vec create_vector(int n_length) {
  arma::vec result(n_length);
  return result;
}

// Create a field of matrices with the given number of fields, rows, and columns
//
// @param n_fields number of fields
// @param n_rows vector of number of rows in each matrix
// @param n_cols number of columns
// @return a field of matrices
arma::field<arma::mat> create_matrix_field(int n_fields, arma::vec n_rows,
                                           int n_cols) {
  arma::field<arma::mat> result(n_fields);
  for (int i = 0; i < n_fields; i++) {
    result[i] = create_matrix(n_rows[i], n_cols);
  }
  return result;
}

// Create a field of matrices with the given number of fields, rows, and columns
//
// @param n_fields number of fields
// @param n_rows number of rows
// @param n_cols vector of number of columns in each matrix
// @return a field of matrices
arma::field<arma::mat> create_matrix_field(int n_fields, int n_rows,
                                           arma::vec n_cols) {
  arma::field<arma::mat> result(n_fields);
  for (int i = 0; i < n_fields; i++) {
    result[i] = create_matrix(n_rows, n_cols[i]);
  }
  return result;
}

// Create a field of vectors with the given number of fields and lengths
//
// @param n_fields number of fields
// @param n_lengths vector of lengths in each vector
// @return a field of vectors
arma::field<arma::vec> create_vector_field(int n_fields, arma::vec n_lengths) {
  arma::field<arma::vec> result(n_fields);
  for (int i = 0; i < n_fields; i++) {
    result[i] = create_vector(n_lengths[i]);
  }
  return result;
}

// Convert a pair of integers to a string
//
// @param i first integer
// @param j second integer
// @return a string representation of the pair, "i.j"
String convert_pair_to_string(int i, int j) {
  return std::to_string(i) + "." + std::to_string(j);
}

// Calculate the log-sum-exp of a vector (Native R)
//
// @param x a vector
// @return the log-sum-exp of the vector
double log_sum_exp(NumericVector x) {
  double lse = 0.0;
  double max_x = max(x);
  for (int i = 0; i < x.size(); i++) {
    lse += exp(x(i) - max_x);
  }
  return log(lse) + max_x;
}

// Calculate the log-sum-exp of a row vector (Armadillo)
//
// @param x a row vector
// @return the log-sum-exp of the row vector
double log_sum_exp(arma::rowvec x) {
  double lse = 0.0;
  double max_x = max(x);
  for (size_t i = 0; i < x.size(); i++) {
    lse += exp(x(i) - max_x);
  }
  return log(lse) + max_x;
}

// Compute the softmax of a vector (Native R)
//
// @param x a vector
// @param log bool, if true, return the log-softmax
// @return the softmax of the vector
NumericVector log_softmax(NumericVector x, bool log) {
  double lse = log_sum_exp(x);
  NumericVector result(x.size());
  for (size_t i = 0; i < x.size(); i++) {
    result(i) = x(i) - lse;
    if (!log) {
      result(i) = exp(result(i));
    }
  }
  return result;
}

// Compute the softmax of a row vector (Armadillo)
//
// @param x a row vector
// @param log bool, if true, return the log-softmax
// @return the softmax of the row vector
arma::rowvec log_softmax(arma::rowvec x, bool log) {
  double lse = log_sum_exp(x);
  arma::rowvec result(x.n_elem);
  for (size_t i = 0; i < x.n_elem; i++) {
    result(i) = x(i) - lse;
    if (!log) {
      result(i) = exp(result(i));
    }
  }
  return result;
}

// Clip a ratio to be between 0 and 1
// If |a - b| < tol, return a value close to 1 (1 - tol)
// If |a / b| < tol, return a value close to 0 (tol)
// Otherwise, return a / b
//
// @param a numerator
// @param b denominator
// @param tol tolerance
// @return the clipped ratio
double clip_ratio(double a, double b, double tol) {
  if (std::fabs(a - b) < tol) {
    return (1.0 - tol);
  } else if (std::fabs(a / b) < tol) {
    return tol;
  } else {
    return a / b;
  }
}

// Check if a list has a specific name
//
// @param x a list
// @param name a string
// @return true if the list has the name, false otherwise
bool has_name(List x, std::string name) {
  CharacterVector nms = x.names();
  return std::find(nms.begin(), nms.end(), name) != nms.end();
}

// Join a list of integer vectors into a single integer vector
//
// @param vectors a field of integer vectors
// @param k number of files
// @param N total number of records, sum(n)
// @return a single integer vector
arma::ivec append_vectors(arma::field<arma::ivec> vectors, int k, int N) {
  arma::ivec result(N, arma::fill::zeros);

  int current_pos = 0;
  for (int i = 0; i < k; i++) {
    arma::ivec current_vec = vectors(i);
    result.subvec(current_pos, current_pos + current_vec.n_elem - 1) =
        current_vec;
    current_pos += current_vec.n_elem;
  }

  return result;
}

// Find duplicate indexes in a vector
//
// @param vec a vector
// @return a field of vectors with the duplicate indexes
arma::field<arma::ivec> find_duplicates(const arma::ivec& vec) {
  std::set<int> processed;
  std::vector<arma::ivec> temp_result;

  for (size_t i = 0; i < vec.n_elem; i++) {
    if (processed.find(vec(i)) == processed.end()) {
      arma::uvec indices = arma::find(vec == vec(i));
      if (indices.n_elem > 1) {
        arma::ivec int_indices = arma::conv_to<arma::ivec>::from(indices);
        temp_result.push_back(int_indices);
      }
      processed.insert(vec(i));
    }
  }

  arma::field<arma::ivec> result(temp_result.size());
  for (size_t i = 0; i < temp_result.size(); i++) {
    result(i) = temp_result[i];
  }

  return result;
}

/******************************************************************************
    Record Linkage:
        Functions commonly used when performing record linkage tasks.
******************************************************************************/
// Initialize the true latent values y_j'
// using the given initial linkage structure
//
// @param lambda_init pre-specified linkage structure
// @param x_append appended data
// @param k number of files
// @param p number of common fields
// @param N total number of records, sum(n)
// @return initialized y, N by p matrix
arma::mat init_true_latent(const arma::field<arma::vec>& lambda_init,
                           const arma::mat& x_append, int k, int p, int N) {
  arma::mat y(N, p);

  arma::vec lambda_append;
  for (int i = 0; i < k; i++) {
    lambda_append = arma::join_cols(lambda_append, lambda_init(i));
  }

  for (int jprime = 0; jprime < N; jprime++) {
    arma::uvec idx = arma::find(lambda_append == jprime);
    if (idx.n_elem == 0) {
      y.row(jprime) = x_append.row(sample(N - 1, 1)(0));
    } else if (idx.n_elem == 1) {
      y.row(jprime) = x_append.row(idx(0));
    } else {
      for (int l = 0; l < p; l++) {
        int chosen = sample_integer(idx);

        y(jprime, l) = x_append(chosen, l);
      }
    }
  }

  return y;
}

// Initialize the distortion field of matrices z
// using the given initial linkage structure and true latent values
//
// @param x original data
// @param y true latent values
// @param lambda_init pre-specified linkage structure
// @param k number of files
// @param n vector of number of records in each file
// @param p number of common fields
// @param discrete_fields indices of discrete fields
// @param n_discrete_fields number of discrete fields
// @param continuous_fields indices of continuous fields
// @param n_continuous_fields number of continuous fields
// @param hyper_delta range of fuzziness (discrete fields)
// @param hyper_tau range of fuzziness (continuous fields)
// @return initialized z, k field with n by p matrices
arma::field<arma::mat> init_distortion(
    const arma::field<arma::mat>& x, const arma::mat& y,
    const arma::field<arma::vec>& lambda_init, int k, arma::vec n, int p,
    arma::vec discrete_fields, int n_discrete_fields,
    arma::vec continuous_fields, int n_continuous_fields, arma::vec hyper_delta,
    arma::vec hyper_tau) {
  arma::field<arma::mat> z(k);

  for (int i = 0; i < k; i++) {
    arma::mat z_i(n[i], p, arma::fill::zeros);

    for (int j = 0; j < n[i]; j++) {
      int jprime = lambda_init(i)(j);
      for (int li = 0; li < n_discrete_fields; li++) {
        int l = discrete_fields[li];
        z_i(j, l) =
            (int)(std::abs(y(jprime, l) - x(i)(j, l)) > hyper_delta[li]);
      }

      for (int li = 0; li < n_continuous_fields; li++) {
        int l = continuous_fields[li];
        z_i(j, l) = (int)(std::abs(y(jprime, l) - x(i)(j, l)) >
                          (2.0 * std::sqrt(hyper_tau[l])));
      }
    }
    z(i) = z_i;
  }

  return z;
}

// Find a list of identically matching records
//
// @param x original data
// @param N total number of records, sum(n)
// @param fields indexes of fields to be considered
// @param linked_only bool, false -> return not-linked records too
// @return list of pairs (vectors) of identical matching records
arma::field<arma::ivec> identical_matching(arma::mat x, int N, arma::vec fields,
                                           bool linked_only) {
  std::vector<std::vector<int>> matches;
  std::vector<bool> visited(N, false);

  int n_fields = fields.n_elem;

  for (int i = 0; i < N; i++) {
    if (!visited[i]) {
      std::vector<int> match;
      match.push_back(i);
      visited[i] = true;

      for (int j = i + 1; j < N; j++) {
        bool identical = true;
        for (int li = 0; li < n_fields; li++) {
          int l = fields[li];
          if (x(i, l) != x(j, l)) {
            identical = false;
            break;
          }
        }

        if (identical) {
          match.push_back(j);
          visited[j] = true;
        }
      }

      if (linked_only) {
        if (match.size() > 1) {
          matches.push_back(match);
        }
      } else {
        matches.push_back(match);
      }
    }
  }

  arma::field<arma::ivec> result(matches.size());
  for (long unsigned int i = 0; i < matches.size(); i++) {
    result(i) = arma::ivec(matches[i]);
  }

  return result;
}

// Sample linkages that contain certin records
// Suppose x_1 and x_2 are linked, but if `1` is not in selected,
// then `1` will be removed from the given linkage structure
//
// @param selected vector of record indexes to retain linkage
// @param identical_links entire linkage structure
// @return sample_identical_links sampled linkage contains selected records
arma::field<arma::ivec> sample_links(IntegerVector selected,
                                     arma::field<arma::ivec> identical_links) {
  // Find identical links with given records indexes
  std::vector<std::vector<int>> matches;
  for (unsigned int i = 0; i < identical_links.n_elem; i++) {
    IntegerVector linked_i_old = as<IntegerVector>(wrap(identical_links(i)));
    IntegerVector linked_i;
    for (int j = 0; j < linked_i_old.length(); j++) {
      IntegerVector idx = {linked_i_old(j)};
      if (in(idx, selected)(0)) {
        linked_i.push_back(idx(0));
      }
    }
    if (linked_i.length() > 1) {
      matches.push_back(as<std::vector<int>>(wrap(linked_i)));
    }
  }

  arma::field<arma::ivec> result(matches.size());
  for (unsigned int i = 0; i < matches.size(); i++) {
    result(i) = arma::ivec(matches[i]);
  }

  return result;
}

// Get initial linkage structure (Lambda) based on the sample linkage structure
//
// @param link_sample a randomly sampled linkage structure
// @param index_matrix a matrix with records indexes
// @param k number of files
// @param n vector of number of records in each file
// @param N total number of records, sum(n)
// @return initial_lambda
arma::field<arma::vec> get_initial_lambda(arma::field<arma::ivec> link_sample,
                                          arma::mat index_matrix, int k,
                                          arma::vec n, int N) {
  arma::field<arma::vec> lambda_init(k);
  for (int i = 0; i < k; i++) {
    arma::vec lambda_i(n(i));
    lambda_init(i) = lambda_i;
  }

  // Construct Lambda^Identical
  std::vector<bool> visited_lambda(N, false);
  for (unsigned int ls = 0; ls < link_sample.n_elem; ls++) {
    arma::ivec link_jprime = link_sample(ls);

    for (unsigned int lambda = 0; lambda < link_jprime.n_elem; lambda++) {
      int jprime = link_jprime(lambda);

      lambda_init(index_matrix(jprime, 0))(index_matrix(jprime, 1)) = ls;
      visited_lambda[jprime] = true;
    }
  }

  // The remaining records are considered as not linked at all
  int jprime = link_sample.n_elem;
  for (int ij = 0; ij < N; ij++) {
    if (!visited_lambda[ij]) {
      lambda_init(index_matrix(ij, 0))(index_matrix(ij, 1)) = jprime;
      jprime += 1;
    }
  }

  return lambda_init;
}

// Generate initial linkage structure based on identical matching
// Among all identical matched links, it is constructed by sampling
// p_init * 100% of links
//
// @param x original data
// @param k number of files
// @param n vector of number of records in each file
// @param N total number of records, sum(n)
// @param non_fuzzy_fields indexes of non-fuzzy fields
// @param index_matrix a matrix with records indexes
// @param p_init sampling proportion [0, 1]
// @param initial_values pre-specified linkage structure
// @return initialized_linkage_index
arma::field<arma::vec> initial_linkage(arma::mat x, int k, arma::vec n, int N,
                                       arma::vec non_fuzzy_fields,
                                       arma::mat index_matrix, double p_init,
                                       Nullable<List> initial_values) {
  arma::field<arma::vec> lambda_init(k);
  for (int i = 0; i < k; i++) {
    arma::vec lambda_i(n(i));
    lambda_init(i) = lambda_i;
  }

  arma::field<arma::ivec> ilink_list;
  if (initial_values.isNull()) {
    // Find identical records
    ilink_list = identical_matching(x, N, non_fuzzy_fields);
  } else {
    List custom_linkage;
    custom_linkage = initial_values;

    arma::field<arma::ivec> custom_lambda = custom_linkage["linkage"];
    for (int i = 0; i < k; i++) {
      custom_lambda(i) -= 1;
    }

    arma::ivec appended_lambda = append_vectors(custom_lambda, k, N);

    ilink_list = find_duplicates(appended_lambda);
  }

  // #(sample links)
  int n_sample = int(p_init * ilink_list.n_elem);

  // Sample links from the identical links
  arma::uvec selected = sample_index(ilink_list.n_elem, n_sample);
  arma::field<arma::ivec> ilink_samples(n_sample);
  for (int s = 0; s < n_sample; s++) {
    ilink_samples(s) = ilink_list(selected(s));
  }

  lambda_init = get_initial_lambda(ilink_samples, index_matrix, k, n, N);

  return lambda_init;
}

// Calculate the posterior similarity matrix
// using the parameters 'nu' of approximated posterior
// obtained from VI (or EVIL) CHOMPER
// It returns either the symmetric matrix
// or the long-form matrix with the indexes of pairs
//
// @param probs_field a field of matrices with posterior probabilities
// @param symmetric bool, if true, return the symmetric matrix
// @return result posterior similarity of all possible pairs
arma::mat posterior_similarity(arma::field<arma::mat> probs_field,
                               bool symmetric) {
  arma::mat probs = compile_matrix(probs_field);
  int n = probs.n_rows;

  arma::mat result;
  if (symmetric) {
    result = probs * probs.t();
    for (int i = 0; i < n; i++) {
      result(i, i) = 1.0;
    }
  } else {
    arma::mat combinations = create_combination(n, 1);

    arma::mat similarity(combinations.n_rows, 1);
    int row = 0;
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        similarity(row++, 0) = sum(probs.row(i) % probs.row(j));
      }
    }

    result = join_horiz(combinations, similarity);
  }
  return result;
}

// Calculate the posterior similarity matrix
// using the MCMC samples of 'lambda' from MCMC-CHOMPER
// It returns the symmetric matrix
//
// @param samples an N by nmcmc matrix with MCMC samples
// @return result posterior similarity of all possible pairs
arma::mat posterior_similarity(arma::mat samples) {
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

/******************************************************************************
    Random Number Generator:
        Functions of random number generators that do not exist in R or Rcpp.
******************************************************************************/
// Generate a random number from a Bernoulli distribution
//
// @param p probability of success
// @return a random number (0 or 1) from a Bernoulli distribution
int rbernoulli(double p) {
  if (R::runif(0.0, 1.0) < p) {
    return 1;
  } else {
    return 0;
  }
}

// Generate a random number from a Dirichlet distribution
//
// @param alpha a vector of parameters
// @return a random number from a Dirichlet distribution
NumericVector rdirichlet(NumericVector alpha) {
  int length = alpha.size();
  NumericVector res(length);
  for (int i = 0; i < length; i++) {
    res(i) = R::rgamma(alpha[i], 1.0);
  }
  res = res / sum(res);
  return res;
}

// Generate a random number from a categorical distribution
// The size of 'probs' and 'size' must be the same
//
// @param probs a vector of probabilities
// @param size number of categories
// @return a random number (category index) from a categorical distribution
int rcategorical(NumericVector probs, int size) {
  IntegerVector x = seq(1, size);
  IntegerVector sample_vec = sample(x, 1, false, probs);
  return sample_vec(0);
}

// Generate a random number from an inverse gamma distribution
//
// @param shape shape parameter
// @param scale scale parameter
// @return a random number from an inverse gamma distribution
double rinvgamma(double shape, double scale) {
  return 1.0 / R::rgamma(shape, 1.0 / scale);
}

// Calculate a density from a Gaussian distribution
//
// @param x value for the density
// @param mu mean
// @param sd standard deviation
// @param log if true, returns log(Gaussian(x))
// @return a density at x
double gaussian_pdf(double x, double mu, double sd, bool use_log) {
  double INV_SQRT_2PI = 0.3989422804014327;
  double z = (x - mu) / sd;
  if (use_log) {
    return log(INV_SQRT_2PI / sd) - 0.5 * z * z;
  } else {
    return (INV_SQRT_2PI / sd) * exp(-0.5 * z * z);
  }
}