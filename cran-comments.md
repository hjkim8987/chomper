## Update (0.1.4)

This update addresses request to make linking to libatomic conditional on its availability.

* The unconditional use of -latomic in src/Makevars has been replaced with a check for library availability.
* Updated the version number to 0.1.4.

## R CMD check results

0 errors | 0 warnings | 1 note

## Test environments

* local macOS (Apple Silicon), R 4.4.1
* GitHub Actions: macos-latest (release), macos-latest (devel), windows-latest (release), ubuntu-latest (devel), ubuntu-latest (release), ubuntu-latest(oldrel-1)
* win-builder: R-devel, R-release
* Note: rhub checks failed due to outdated Rust version in rhub environments (`salso` dependency requires Rust >= 1.80.1)