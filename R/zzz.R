# global references (will be initialized in .onLoad)
transformers <- NULL
torch <- NULL
# data table :=
.datatable.aware <- TRUE

#' @noRd
.onLoad <- function(libname, pkgname) {

  # This will instruct reticulate to immediately try to configure the
  # active Python environment, installing any required Python packages
  # as necessary.
  reticulate::configure_environment(pkgname)
  # use superassignment to update global reference
  transformers <<- reticulate::import("transformers",
    delay_load = TRUE,
    convert = FALSE
  )
  torch <<- reticulate::import("torch", delay_load = TRUE, convert = FALSE)
  # TODO message or something if it's not installed
  op <- options()
  op.pangoling <- list(
    pangoling.debug = FALSE,
    pangoling.verbose = 2,
    pangoling.cache = cachem::cache_mem(max_size = 1024 * 1024^2),
    pangoling.causal.default = "gpt2",
    pangoling.masked.default = "bert-base-uncased"
  )
  toset <- !(names(op.pangoling) %in% names(op))
  if (any(toset)) options(op.pangoling[toset])

  # caching:
  tokenizer <<- memoise::memoise(tokenizer)
  lang_model <<- memoise::memoise(lang_model)
  transformer_vocab <<- memoise::memoise(transformer_vocab)

  # avoid notes:
  utils::globalVariables(c("mask_n"))

  invisible()
}

.onAttach <- function(libname, pkgname) {
    packageStartupMessage(pkgname, " version ", packageVersion(pkgname),"\n Pretrained models and tokenizers are downloaded from https://huggingface.co/ the first time they are used. For changing the cache folder use:\n
                         set_cache_folder(my_new_path)")
}
## .onAttach <- function(libname, pkgname) {
##    #only once per session
##   if (env_has(inform_env, id)) {
##     return(invisible(NULL))
##   }
##   inform_env[[id]] <- TRUE
##  ()
## }

