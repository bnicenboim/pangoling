# global references (will be initialized in .onLoad)
transformers <- NULL
torch <- NULL
# data table :=
.datatable.aware <- TRUE

mybib <- RefManageR::ReadBib(system.file("REFERENCES.bib", package="pangoling"), check = FALSE)


#' @noRd
.onLoad <- function(libname, pkgname) {
  # This will instruct reticulate to immediately try to configure the active Python environment, installing any required Python packages as necessary.
  reticulate::configure_environment(pkgname)
  # use superassignment to update global reference
    transformers <<- reticulate::import("transformers", delay_load = TRUE, convert = FALSE)
    torch <<- reticulate::import("torch", delay_load = TRUE, convert = FALSE)
    #TODO message or something if it's not installed
    op <- options()
    op.pangoling <- list(
      pangoling.debug = FALSE,
      pangoling.verbose = 2,
      pangoling.cache = cachem::cache_mem(max_size = 1024 * 1024^2)
    )
    toset <- !(names(op.pangoling) %in% names(op))
    if (any(toset)) options(op.pangoling[toset])
    invisible()
    }
