#' @noRd
require_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste0("Package '", pkg, "'  needed for this function to work. Please install it."),
         call. = FALSE
    )
  }
}

#' @noRd
`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) {
    y
  } else {
    x
  }
}
#' @noRd
message_verbose <- function(...) {
  if (options()$pangolang.verbose) message(...)
}
