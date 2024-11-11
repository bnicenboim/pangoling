#' @keywords internal
#'
#' @details
#' These options are used to control various aspects of the pangoling package.
#' Users can customize these options via the `options()` function by specifying `pangoling.<option>` names.
#' - `pangoling.debug`: Logical; when `TRUE`, enables debugging mode. Default is `FALSE`.
#' - `pangoling.verbose`: Integer; controls the verbosity level (e.g., 0 = silent, 1 = minimal, 2 = detailed). Default is `2`.
#' - `pangoling.log.p`: Logical; if `TRUE`, probabilities are logged. Default is `TRUE`.If TRUE (default),  x are assumed to be log-transformed probabilities with base e, if FALSE x are assumed to be raw probabilities, alternatively log.p can be the base of other logarithmic transformations.
#' - `pangoling.cache`: A cache object created with `cachem::cache_mem`, allowing you to specify the maximum size (in bytes) for cached objects. Default is `1024 * 1024^2` bytes (1 MB).
#' - `pangoling.causal.default`: Character string; specifies the default model for causal language processing. Default is `"gpt2"`.
#' - `pangoling.masked.default`: Character string; specifies the default model for masked language processing. Default is `"bert-base-uncased"`.
#'
#' Use `options(pangoling.<option> = <value>)` to set these options in your session.
#'
"_PACKAGE"


## usethis namespace: start
#' @importFrom memoise memoise
## usethis namespace: end
NULL
