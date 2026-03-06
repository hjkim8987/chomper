## Resubmission

This is a resubmission. Changes since the first submission:

* Added `-latomic` to `src/Makevars` to fix a linking error on Linux: `__atomic_compare_exchange`.
* Added `inst/WORDLIST` to suppress spell check notes for domain-specific terms (CAVI, Variational).
* Updated the version number to 0.1.1.

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.

## Test environments

* local macOS (Apple Silicon), R 4.4.1
* GitHub Actions: macos-latest (release), windows-latest (release), ubuntu-latest (devel), ubuntu-latest (release), ubuntu-latest(oldrel-1)
* win-builder: R-devel, R-release
* Note: rhub checks failed due to outdated Rust version in rhub environments (`salso` dependency requires Rust >= 1.80.1)