## Resubmission (0.1.3)

This is a resubmission. Changes since previous submission:

* References: I have removed the references from the descriptions. This package implements an original method that has not been published yet, so no references (DOI or URL) are available.
* Replaced `print()` calls with `warning()` in R/inference.R to allow users to suppress console output, as requested.
* Rcpp codes are update to ensure that all console outputs are wrapped in a `verbose` check.
* Updated the version number to 0.1.3.

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.

## Test environments

* local macOS (Apple Silicon), R 4.4.1
* GitHub Actions: macos-latest (release), windows-latest (release), ubuntu-latest (devel), ubuntu-latest (release), ubuntu-latest(oldrel-1)
* win-builder: R-devel, R-release
* Note: rhub checks failed due to outdated Rust version in rhub environments (`salso` dependency requires Rust >= 1.80.1)