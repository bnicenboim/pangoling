# global reference to scipy (will be initialized in .onLoad)
minicons <- NULL

.onLoad <- function(libname, pkgname) {
  # use superassignment to update global reference
    minicons <<- reticulate::import("minicons", delay_load = TRUE, convert = FALSE)
    #TODO message or something if it's not installed
    op <- options()
    op.pangolang <- list(
      pangolang.verbose = TRUE
    )
    toset <- !(names(op.pangolang) %in% names(op))
    if (any(toset)) options(op.pangolang[toset])

    invisible()

    }
