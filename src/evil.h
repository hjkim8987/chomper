#ifndef EVIL_H
#define EVIL_H

#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

#include "cavi.h"

using namespace Rcpp;

class OptimMember {
 private:
  // Data Information
  arma::field<arma::mat> x;  // Original Data

  // File Information
  int k;            // Number of files
  arma::vec n;      // Number of records in each file
  int N;            // Number of total records; sum_i n_i
  double N_double;  // (double) Number of total records; sum_i n_i

  // Field Information
  int p;  // Number of fields in each file (n_discrete + n_continuous)

  arma::vec discrete_fields;  // Indices of discrete fields
  int n_discrete_fields;      // Number of discrete fields
  arma::vec M;  // Number of levels of each field (length n_discrete)
  arma::vec continuous_fields;  // Indices of continuous fields
  int n_continuous_fields;      // Number of continuous fields

  // Hyperparameter
  arma::mat hyper_beta;             // Beta for distortion ratio
  arma::field<arma::vec> hyper_mu;  // Dirichlet for the multinomial theta
  arma::mat hyper_sigma;  // Inverse-Gamma for variance of latent true values
  arma::vec hyper_phi;    // multiplier for the likelihood (length n_discrete)
  arma::vec hyper_tau;    // temperature parameter (length p)
  arma::vec hyper_delta;  // range of discrete fuzziness (length n_discrete)

  double p_init;  // Ratio of the initial linkages

  arma::field<arma::vec> lambda_init;  // Initial linkage indexes

  arma::field<arma::vec> alpha;  // q_theta ~ Dirichlet(alpha)
  arma::field<arma::vec> psi;    // E[log theta_l]

  arma::vec eta_mean;  // q_eta ~ N(mean[l], var[l])
  arma::vec eta_var;

  arma::vec sigma_shape;  // q_sigma ~ IG(shape[l], scale[l])
  arma::vec sigma_scale;

  arma::mat omega;  // q_beta ~ Beta(omega[l, 0], omega[l, 1])
  arma::mat kappa;  // E[log beta_l]

  arma::field<arma::mat> nu;      // q_lambda ~ MN(1, nu)
  arma::field<arma::mat> log_nu;  // log(nu)

  arma::field<arma::mat> gamma;      // q_y ~ MN(1, gamma) (l = 1, ... l_1)
  arma::field<arma::mat> log_gamma;  // log(gamma)

  arma::mat eta_tilde;  // q_y ~ N(eta_tilde[l, j'], sigma_tilde[l, j'])
  arma::mat sigma_tilde;

  arma::field<arma::mat> chi_discrete;             // chi_discrete = gamma * nu
  arma::field<arma::mat> chi_expectation;          // nu * E[y_j'l]
  arma::field<arma::mat> chi_squared_expectation;  // nu * E[y_j'l^2]

  arma::mat zeta_sigma;  // (E[eta_l], E[eta_l^2]) with eta_tilde(t)
  arma::mat zeta_rho;    // (E[eta_l], E[eta_l^2]) with eta_tilde(t+1)

  arma::field<arma::mat> rho;  // q_z ~ Bernoulli(rho)
  arma::mat rho_sum;           // sum_ij E[z_ijl], sum_ij E[1 - z_ijl]

  arma::vec log_phi_tau;  // log(phi^(1 / tau))

  arma::mat x_index;
  arma::mat x_append;

  arma::field<arma::mat> log_constant_matrix;

  double elapsed_time = 0.0;

  int niter_cavi = 0;
  bool interrupted = false;

  double elbo = R_NegInf;

 public:
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
  OptimMember(arma::field<arma::mat> x_, int k_, arma::vec n_, int N_, int p_,
              arma::vec discrete_fields_, int n_discrete_fields_, arma::vec M_,
              arma::vec continuous_fields_, int n_continuous_fields_,
              arma::mat hyper_beta_, arma::field<arma::vec> hyper_mu_,
              arma::mat hyper_sigma_, arma::vec hyper_phi_,
              arma::vec hyper_tau_, arma::vec hyper_delta_);
  ~OptimMember();

  /*******************************************
      1. Constructor and Destructor
    2. Getter and Setter
      3. Functions for CAVI optimization
  *******************************************/
  int getNumberOfFiles() const;
  void setNumberOfFiles(int k_);

  arma::vec getNumberOfRecords() const;
  void setNumberOfRecords(arma::vec n_);

  int getNumberOfTotalRecords() const;
  void setNumberOfTotalRecords(int N_);

  int getNumberOfFields() const;
  void setNumberOfFields(int p_);

  arma::vec getDiscreteFields() const;
  void setDiscreteFields(arma::vec discrete_fields_);

  int getNumberOfDiscreteFields() const;
  void setNumberOfDiscreteFields(int n_discrete_fields_);

  arma::vec getNumberOfLevels() const;
  void setNumberOfLevels(arma::vec M_);

  arma::vec getContinuousFields() const;
  void setContinuousFields(arma::vec continuous_fields_);

  int getNumberOfContinuousFields() const;
  void setNumberOfContinuousFields(int n_continuous_fields_);

  arma::field<arma::mat> getLogConstantMatrix() const;
  void setLogConstantMatrix(arma::field<arma::mat> log_constant_matrix_);

  double getLogNormalizeConstant(int i, int j, int l) const;
  void setLogNormalizeConstant(int i, int j, int l, double value);

  arma::field<arma::vec> getApproximatedAlpha() const;
  void setApproximatedAlpha(arma::field<arma::vec> alpha_);

  arma::field<arma::vec> getApproximatedPsi() const;
  void setApproximatedPsi(arma::field<arma::vec> psi_);

  arma::vec getApproximatedEtaMean() const;
  void setApproximatedEtaMean(arma::vec eta_mean_);

  arma::vec getApproximatedEtaVar() const;
  void setApproximatedEtaVar(arma::vec eta_var_);

  arma::vec getApproximatedSigmaShape() const;
  void setApproximatedSigmaShape(arma::vec sigma_shape_);

  arma::vec getApproximatedSigmaScale() const;
  void setApproximatedSigmaScale(arma::vec sigma_scale_);

  arma::mat getApproximatedOmega() const;
  void setApproximatedOmega(arma::mat omega_);

  arma::mat getApproximatedKappa() const;
  void setApproximatedKappa(arma::mat kappa_);

  arma::field<arma::mat> getApproximatedNu() const;
  void setApproximatedNu(arma::field<arma::mat> nu_);

  arma::field<arma::mat> getApproximatedLogNu() const;
  void setApproximatedLogNu(arma::field<arma::mat> log_nu_);

  arma::field<arma::mat> getApproximatedGamma() const;
  void setApproximatedGamma(arma::field<arma::mat> gamma_);

  arma::field<arma::mat> getApproximatedLogGamma() const;
  void setApproximatedLogGamma(arma::field<arma::mat> log_gamma_);

  arma::mat getApproximatedEtaTilde() const;
  void setApproximatedEtaTilde(arma::mat eta_tilde_);

  arma::mat getApproximatedSigmaTilde() const;
  void setApproximatedSigmaTilde(arma::mat sigma_tilde_);

  arma::field<arma::mat> getApproximatedRho() const;
  void setApproximatedRho(arma::field<arma::mat> rho_);

  arma::mat getApproximatedRhoSum() const;
  void setApproximatedRhoSum(arma::mat rho_sum_);

  double getSamplingProb() const;
  void setSamplingProb(double prob);

  double getElapsedTime() const;
  void setElapsedTime(double elapsed_time_);

  double getELBO() const;
  void setELBO(double elbo_);

  bool isInterrupted() const;
  void setInterrupted(bool interrupted_);

  int getNumberOfIterations() const;
  void setNumberOfIterations(int niter_cavi_);

  /*******************************************
      1. Constructor and Destructor
      2. Getter and Setter
    3. Functions for CAVI optimization
  *******************************************/
  void sum_rho_by_ij();
  void initialize(bool custom_initializer = false,
                  Nullable<List> initial_values = R_NilValue);
  void coordinate_ascent();
  void cavi(double eps, int max_iter, bool verbose, double max_time = 604800);
  void update_mutated_info();
  void mutation_split_and_merge();
};

/******************************************************************************
    Functions for Evolutionary Algorithm:
        A collection of functions handling a vector of OptimMember.
        These are not member fuctions of OptimMember class,
        because they uses multiple OptimMember's.
******************************************************************************/
std::vector<OptimMember> initialization_step(
    OptimMember data, int n_parents, double overlap_prob,
    bool custom_initializer = false,
    Nullable<List> initial_values = R_NilValue);

std::vector<OptimMember> update_step(std::vector<OptimMember> P, double eps,
                                     int max_iter, bool verbose, int n_threads);

std::vector<OptimMember> crossover_step(std::vector<OptimMember> P,
                                        arma::mat combination,
                                        int n_combination, int n_children);

std::vector<OptimMember> selection_step(std::vector<OptimMember>& P1,
                                        std::vector<OptimMember>& P3,
                                        int n_parents, int n_children);

std::vector<OptimMember> mutation_step(std::vector<OptimMember> P4,
                                       int n_parents, int n_threads);

#endif