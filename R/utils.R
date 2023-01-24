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
versions <- function(){
  df_packages <- reticulate::py_list_packages()
  ver <- reticulate::py_version()
  print(paste0("Python version ", ver))
  print("Following version packages. (For the entire list use `reticulate::py_list_packages()`).")
  rel_packages <- df_packages |> filter.(package %in% c("transformers","torch"))
  print(rel_packages)
  invisible(list(python= ver, rel_packages = rel_packages))
}

stop2 <- function (...){
  stop(..., call. = FALSE)
}

#' @noRd
collapse_comma <- function (...){
  paste0("'", ..., "'", collapse = ", ")
}

#' @noRd
list_fields <- function(data) cat(paste0("#' * `",colnames(data), "`: DESCRIPTION\n"))


#' From https://github.com/dapperstats/gendrendr
#' @noRd
get_lang_locale <- function(){
  ismac <- Sys.info()["sysname"] == "Darwin"
  issolaris <- Sys.info()["sysname"] == "SunOS"
  splitchar <- ifelse(ismac | issolaris, "/", ";")
  locale <- Sys.getlocale()
  locale <- strsplit(locale, splitchar)[[1]]
  locale <- sub(".*=", "", locale)
  locale <- strsplit(locale, "_")[[1]]
  locale <- setNames(locale, c("language", "location"))
  locale[["location"]] <- sub("\\..*", "", locale[["location"]])
  locale
}

#' @noRd
writeRDS <- function(object, filename, choice = NULL, ...){
  writable <- file.access(rappdirs::user_data_dir(),
                          mode = 2) == 0
  data_dir <- rappdirs::user_data_dir("pangoling")

  choices <- c(if (writable)
    paste0("Store in ",data_dir," directory."),
    "Store in the temporary directory.",
    "Do not store.")

  if (interactive()){
    choice <- menu(choices, title = paste0("Do you want to store the data set?"))
  } else {
    choice <- if(writable & choice ==1) 1L else 2L
  }
  if (choice == 0 | choice == length(choices))
    return()
  if(choice ==1){
    if(!dir.exists(data_dir)) dir.create(data_dir)
    saveRDS(object,  file.path(data_dir,filename), ...)
  } else {
    saveRDS(object,  file.path(tempdir,filename), ...)
  }

}


download_dataset <- function(url, filename = basename(url), name =""){

  writable <- file.access(rappdirs::user_data_dir(),
              mode = 2) == 0
   DATA_DIR <- rappdirs::user_data_dir("pangoling")

    choices <- c(if (writable)
      paste0("Download to ",DATA_DIR," directory"),
      "Download to the temporary directory",
      "Do not download")

    if (interactive()){
      choice <- menu(choices, title = paste0("Do you want to download the data set, ",name,"from [",url,"]?"))
    } else {
      choice <- if (writable) 1L else 2L
    }
      if (choice == 0 | choice == length(choices))
        stop("Cannot proceed without the dataset")
    if(choice ==1){
      if(!dir.exists(DATA_DIR)) dir.create(DATA_DIR)
      FILE <- file.path(DATA_DIR,filename)
    } else {
      FILE <- file.path(tempdir(),filename)
    }


    httr::GET(url,
        httr::write_disk(FILE, overwrite = TRUE),
        httr::progress()
    )
}

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
  if (options()$pangoling.verbose > 0) message(...)
}

#' @noRd
message_debug <- function(...) {
  if (options()$pangoling.verbose > 1) message(...)
}


log_softmax <- function(x) {
  # 1.4 from https://academic.oup.com/imajna/article/41/4/2311/5893596
  a <- max(x)
  log(exp(x - a)/sum(exp(x-a)))
}

log_softmax2 <- function(x) {
  # 1.5 from https://academic.oup.com/imajna/article/41/4/2311/5893596
  x - matrixStats::logSumExp(x)
}

split_in_sentences <- function(x){
  pot_end <- which(chr_detect(x, "(\\.|\\?|\\!)$"))
  pot_end <- pot_end[chr_detect(x[pot_end+1], "^([A-Z])")]
  tidytable::map2.(c(1, pot_end+1), c(pot_end, length(x)), ~ x[.x:.y])
}

