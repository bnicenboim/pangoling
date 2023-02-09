#' Calculates perplexity
#'
#' Calculates perplexity of a vector of (log-)probabilities.
#'
#' If x are raw probabilities (NOT the default), then perplexity is calculated as follows:
#'
#' \deqn{\left(\prod_{n} x_n \right)^\frac{1}{N}}
#'
#' @param x	A vector of log-probabilities.
#' @param na.rm	Should missing values (including NaN) be removed?
#' @param log.p If TRUE (default),  x are assumed to be log-transformed probabilities with base e, if FALSE x are assumed to be raw probabilities, alternatively log.p can be the base of other logarithmic transformations.
#' @return The perplexity.
#'
#' @examples
#' probs <- c(.3, .5, .6)
#' perplexity(probs, log.p = FALSE)
#' lprobs <- log(probs)
#' perplexity(lprobs, log.p = TRUE)
#' @export
#'
perplexity <- function(x, na.rm = FALSE, log.p
                       = TRUE) {
  if (log.p == FALSE) {
    prod(x, na.rm = na.rm)^(-1 / length(x))
  } else if (log.p || all.equal(log.p, exp(1))) {
    exp(-sum(x, na.rm = na.rm) / length(x))
  } else {
    log.p^(-sum(x, na.rm = na.rm) / length(x))
  }
}