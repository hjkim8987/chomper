## Update (0.1.4)

This update addresses the reported CRAN check problems concerning unavailable suggested packages and unconditional use of suggested packages in the vignette.

* Removed 'blink' from Suggests and replaced the uses of blink::links() in the vignette and README with an in-package function that converts cluster labels to linkage structures.
* Made the use of 'salso', 'ggplot2', and 'patchwork' conditional in the vignette, including subsequent chunks that depend on their results.
* Corrected performance() to count true negatives and false negatives over pairs of records, consistently with true positives and false positives.
* Updated the version number to 0.1.4.

## R CMD check results

0 errors | 0 warnings | 0 notes

## Test environments

* local macOS (Apple Silicon), R 4.4.1
* GitHub Actions: macos-latest (release), macos-latest (devel), windows-latest (release), ubuntu-latest (devel), ubuntu-latest (release), ubuntu-latest(oldrel-1)
* win-builder: R-devel, R-release
* Note: rhub checks failed due to outdated Rust version in rhub environments (`salso` dependency requires Rust >= 1.80.1)