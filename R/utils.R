stop2 <- function(...) {
  stop(..., call. = FALSE)
}



#' @noRd
message_verbose <- function(...) {
  if (options()$pangoling.verbose > 0) message(...)
}

#' @noRd
message_debug <- function(...) {
  if (options()$pangoling.verbose > 1) message(...)
}

#' Replacement of str_match
#' @noRd
chr_match <- function(string, pattern) {
  matches <- regexec(pattern = pattern, text = string)
  list_matches <- lapply(
    regmatches(x = string, m = matches),
    function(x) if (length(x) == 0) NA else x
  )
  do.call("rbind", list_matches)
}

#' Replacement of str_detect
#' @noRd
chr_detect <- function(string, pattern, ignore.case = FALSE) {
  grepl(pattern = pattern, x = string, ignore.case = ignore.case)
}


