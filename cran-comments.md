## Resubmission (0.1.2)

This is a resubmission. Changes since previous submission:

* Replaced all instances of `abs()` with `std::abs()` or `std::fabs()` to ensure proper handling of floating-point arguments, resolving the `-Wabsolute-value` warning.
* Updated the version number to 0.1.2.

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new release.

## Test environments

* local macOS (Apple Silicon), R 4.4.1
* GitHub Actions: macos-latest (release), windows-latest (release), ubuntu-latest (devel), ubuntu-latest (release), ubuntu-latest(oldrel-1)
* win-builder: R-devel, R-release
* Note: rhub checks failed due to outdated Rust version in rhub environments (`salso` dependency requires Rust >= 1.80.1)