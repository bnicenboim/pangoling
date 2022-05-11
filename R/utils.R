#' From https://github.com/dapperstats/gendrendr
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

writeRDS <- function(object, filename, choice = NULL, ...){
  writable <- file.access(rappdirs::user_data_dir(),
                          mode = 2) == 0
  data_dir <- rappdirs::user_data_dir("pangolang")

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
   DATA_DIR <- rappdirs::user_data_dir("pangolang")

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
  if (options()$pangolang.verbose) message(...)
}
