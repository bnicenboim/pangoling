# global references (will be initialized in .onLoad)
transformers <- NULL
torch <- NULL
# data table :=
.datatable.aware <- TRUE

#' @noRd
.onLoad <- function(libname, pkgname) {

  reticulate::use_virtualenv("r-pangoling", required = FALSE)

  # use superassignment to update global reference
  transformers <<- reticulate::import("transformers",
    delay_load = TRUE,
    convert = FALSE)
  inspect <<- reticulate::import("inspect", delay_load = TRUE, convert = TRUE) 
  torch <<- reticulate::import("torch", delay_load = TRUE, convert = FALSE)
  # TODO message or something if it's not installed
  # ask about the env
  op <- options()
  op.pangoling <- list(
    pangoling.debug = FALSE,
    pangoling.verbose = 2,
    pangoling.log.p = TRUE,
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
    packageStartupMessage(pkgname, 
    " version ", 
    utils::packageVersion(pkgname),
    "\nAn introduction to the package can be found in https://bruno.nicenboim.me/pangoling/articles/\n Notice that pretrained models and tokenizers are downloaded from https://huggingface.co/ the first time they are used.\n For changing the cache folder use:\n set_cache_folder(my_new_path)")
}
