#' Install the Python packages needed for `pangoling`
#'
#' @description
#' `install_py_pangoling` function facilitates the installation of Python packages needed for using `pangoling` within an R environment,
#' utilizing the `reticulate` package for managing Python environments. It supports various installation methods,
#' environment settings, and Python versions.
#'
#' @usage
#' install_py_pangoling(method = c("auto", "virtualenv", "conda"),
#'                      conda = "auto",
#'                      version = "default",
#'                      envname = "r-pangoling",
#'                      restart_session = TRUE,
#'                      conda_python_version = NULL,
#'                      ...,
#'                      pip_ignore_installed = FALSE,
#'                      new_env = identical(envname, "r-pangoling"),
#'                      python_version = NULL)
#'
#' @param method A character vector specifying the environment management method. 
#'               Options are 'auto', 'virtualenv', and 'conda'. Default is 'auto'.
#' @param conda Specifies the conda binary to use. Default is 'auto'.
#' @param version The Python version to use. Default is 'default', automatically selected.
#' @param envname Name of the virtual environment. Default is 'r-pangoling'.
#' @param restart_session Logical, whether to restart the R session after installation. 
#'                        Default is TRUE.
#' @param conda_python_version Python version for conda environments.
#' @param ... Additional arguments passed to `reticulate::py_install`.
#' @param pip_ignore_installed Logical, whether to ignore already installed packages. 
#'                             Default is FALSE.
#' @param new_env Logical, whether to create a new environment if `envname` is 'r-pangoling'.
#'               Default is the identity of `envname`.
#' @param python_version Specifies the Python version for the environment.
#'
#' @details
#' This function automatically selects the appropriate method for environment management and Python installation,
#' with a focus on virtual and conda environments. It ensures flexibility in dependency management and Python version control.
#' If a new environment is created, existing environments with the same name are removed.
#'
#' @return
#' The function returns `NULL` invisibly, but outputs a message on successful installation.
#' @export
install_py_pangoling <- function(method = c("auto", "virtualenv", "conda"),
                                 conda = "auto",
                                 version = "default", 
                                 envname = "r-pangoling", 
                                 restart_session = TRUE,
                                 conda_python_version = NULL,
                                 ...,
                                 pip_ignore_installed = FALSE,
                                 new_env = identical(envname, "r-pangoling"),
                                 python_version = NULL
                                 ){

  method <- match.arg(method)
  
  python_version <- python_version %||% conda_python_version
  if(method %in% c("auto", "virtualenv") &&
     is.null(python_version)) {
    
    # virtualenv_starter() picks the most recent version available, but older
    # versions of tensorflow typically don't work with the latest Python
    # release. In general, we're better off picking the oldest Python version available
    # that works with the current release of tensorflow.

    available <- reticulate::virtualenv_starter(version = ">=3.9", all = TRUE)
    # pick the smallest minor version, ignoring patchlevel
    if(nrow(available))
      python_version <- min(available$version[, 1:2])
  }
  
  if (isTRUE(new_env)) {
    
    if (method %in% c("auto", "virtualenv") &&
        reticulate::virtualenv_exists(envname))
      reticulate::virtualenv_remove(envname = envname, confirm = FALSE)
    
    if (method %in% c("auto", "conda")) {
      if (!is.null(tryCatch(reticulate::conda_python(envname, conda = conda),
                            error = function(e) NULL)))
        reticulate::conda_remove(envname, conda = conda)
    }
    
  }
  packages <- c("transformers", "torch")
  py_install_args <- list(
    packages       = packages,
    envname        = envname,
    method         = method,
    conda          = conda,
    python_version = python_version,
    pip            = TRUE,
    pip_ignore_installed = pip_ignore_installed,
    ...
  )
  
  do.call(reticulate::py_install, py_install_args)
  cat("\nInstallation complete.\n\n")
  
  if (restart_session &&
      requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::hasFun("restartSession"))
    rstudioapi::restartSession()
  
  invisible(NULL)

}


#' @noRd
message_verbose <- function(...) {
  if (options()$pangoling.verbose > 0) message(...)
}

stop2 <- function(...) {
  stop(..., call. = FALSE)
}

#' #' Replacement of str_match
#' #' @noRd
#' chr_match <- function(string, pattern) {
#'   matches <- regexec(pattern = pattern, text = string)
#'   list_matches <- lapply(
#'     regmatches(x = string, m = matches),
#'     function(x) if (length(x) == 0) NA else x
#'   )
#'   do.call("rbind", list_matches)
#' }


#' Replacement of str_detect
#' @noRd
chr_detect <- function(string, pattern, ignore.case = FALSE) {
  grepl(pattern = pattern, x = string, ignore.case = ignore.case)
}


#' #' @noRd
#' message_debug <- function(...) {
#'   if (options()$pangoling.verbose > 1) message(...)
#' }

"%||%" <- function(x, y) if (is.null(x)) y else x
