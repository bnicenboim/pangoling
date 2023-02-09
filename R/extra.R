#' An alternative to the internal \code{do.call}
#'
#' The \code{\link[base]{do.call}} can be somewhat slow, especially when working
#' with large objects. This function is based upon the suggestions from Hadley
#' Wickham on the R mailing list.
#' Also thanks to \emph{Tommy} at StackOverflow for
#' \href{https://stackoverflow.com/questions/10022436/do-call-in-combination-with}{suggesting}
#' how to handle double and triple colon operators, \code{::}, further
#' enhancing the function.
#'
#' @author https://github.com/gforge/Gmisc/blob/master/R/fastDoCall.R
#'
#' @inheritParams base::do.call
#'
#' @section Note:
#'
#' While the function attempts to do most of what \code{\link[base]{do.call}}
#' can it has limitations. It can currently not parse the example code from the
#' original function: \code{do.call(paste, list(as.name("A"), as.name("B")), quote = TRUE)}
#' and the functionality of \code{quote} has not been thoroughly tested.
#' @noRd
fastDoCall <- function(what, args, quote = FALSE, envir = parent.frame()) {
  if (quote) {
    args <- lapply(args, enquote)
  }

  if (is.null(names(args)) ||
      is.data.frame(args)) {
    argn <- args
    args <- list()
  } else {
    # Add all the named arguments
    argn <- lapply(names(args)[names(args) != ""], as.name)
    names(argn) <- names(args)[names(args) != ""]
    # Add the unnamed arguments
    argn <- c(argn, args[names(args) == ""])
    args <- args[names(args) != ""]
  }

  if ("character" %in% class(what)) {
    if (is.character(what)) {
      fn <- strsplit(what, "[:]{2,3}")[[1]]
      what <- if (length(fn) == 1) {
        get(fn[[1]], envir = envir, mode = "function")
      } else {
        get(fn[[2]], envir = asNamespace(fn[[1]]), mode = "function")
      }
    }
    call <- as.call(c(list(what), argn))
  } else if ("function" %in% class(what)) {
    f_name <- deparse(substitute(what))
    call <- as.call(c(list(as.name(f_name)), argn))
    args[[f_name]] <- what
  } else if ("name" %in% class(what)) {
    call <- as.call(c(list(what, argn)))
  }

  eval(call,
       envir = args,
       enclos = envir
  )
}


#' Show the version of python and relevant packages
#'
#' @return a list with the python version, and the packages needed for `pangoling`
#'
#' @export
#'
#' @examples
#'
#' versions()
#'
#' @noRd
versions <- function() {
  df_packages <- reticulate::py_list_packages()
  ver <- reticulate::py_version()
  print(paste0("Python version ", ver))
  print("Following version packages. (For the entire list use `reticulate::py_list_packages()`).")
  rel_packages <- df_packages |> filter.(package %in% c("transformers", "torch"))
  print(rel_packages)
  invisible(list(python = ver, rel_packages = rel_packages))
}

log_softmax <- function(x) {
  # 1.4 from https://academic.oup.com/imajna/article/41/4/2311/5893596
  a <- max(x)
  log(exp(x - a) / sum(exp(x - a)))
}

log_softmax2 <- function(x) {
  # 1.5 from https://academic.oup.com/imajna/article/41/4/2311/5893596
  x - matrixStats::logSumExp(x)
}


#' @noRd
require_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste0("Package '", pkg, "'  needed for this function to work. Please install it."),
         call. = FALSE
    )
  }
}
