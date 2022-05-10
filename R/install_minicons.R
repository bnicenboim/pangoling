#' Title
#'
#' @param method
#' @param conda
#'
#' @return
#' @export
#'
#' @examples
install_minicons <- function(method = "auto", conda = "auto") {
  reticulate::py_install("minicons", method = method, conda = conda)
}
