# chomper 0.1.5

* Added `links_from_labels()` to convert cluster labels, such as unique identifiers or a point estimate of the partition, into the linkage structure required by `performance()`.
* Removed the 'blink' package from Suggests. The vignette and README now use `links_from_labels()` instead of `blink::links()`.
* Suggested packages ('salso', 'ggplot2', and 'patchwork') are now used conditionally in the vignette.
* Fixed `performance()` to count true negatives and false negatives over pairs of records, consistently with true positives and false positives.

# chomper 0.1.4

* Make linking to libatomic conditional on library availability, improving build portability.