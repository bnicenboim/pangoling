# global references (will be initialized in .onLoad)
minicons <- NULL
## store the datasets here:
.pkg_env <- new.env(parent=emptyenv())
# data table :=
.datatable.aware <- TRUE

.onLoad <- function(libname, pkgname) {
  # use superassignment to update global reference
    minicons <<- reticulate::import("minicons", delay_load = TRUE, convert = FALSE)
    #TODO message or something if it's not installed
    op <- options()
    op.pangolang <- list(
      pangolang.verbose = TRUE,
      pangolang.cache = cachem::cache_mem(max_size = 1024 * 1024^2)
    )
    toset <- !(names(op.pangolang) %in% names(op))
    if (any(toset)) options(op.pangolang[toset])


    invisible()

    }
