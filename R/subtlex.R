#' Get a table with word frequencies.
#'
#' @param x
#' @param source
#' @param language
#' @param location
#' @param ignore_case
#' @param remove_punc
#' @param regex
#'
#' From https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus
#' The word. This starts with a capital when the word more often starts with an uppercase letter than with a lowercase letter.
#' FREQcount. This is the number of times the word appears in the corpus (i.e., on the total of 51 million words).
#' CDcount. This is the number of films in which the word appears (i.e., it has a maximum value of 8,388).
#' FREQlow. This is the number of times the word appears in the corpus starting with a lowercase letter. This allows users to further match their stimuli.
#' CDlow. This is the number of films in which the word appears starting with a lowercase letter.
#' SUBTLWF. This is the word frequency per million words. It is the measure you would preferably use in your manuscripts, because it is a standard measure of word frequency independent of the corpus size. It is given with two digits precision, in order not to lose precision of the frequency counts.
#' Lg10WF. This value is based on log10(FREQcount+1) and has four digit precision. Because FREQcount is based on 51 million words, the following conversions apply for SUBTLEXUS:
#'   Lg10WF 	SUBTLWF
#' 1.00 	0.2
#' 2.00 	2
#' 3.00 	20
#' 4.00 	200
#' 5.00 	2000
#' SUBTLCD indicates in how many percent of the films the word appears. This value has two-digit precision in order not to lose information.
#' Lg10CD. This value is based on log10(CDcount+1) and has four digit precision. It is the best value to use if you want to match words on word frequency. As CDcount is based on 8388 films, the following conversions apply:
#'   Lg10CD 	SUBTLCD
#' 0.95 	0.1
#' 1.93 	1
#' 2.92 	10
#' 3.92 	100
#' @return
#' @export
#'
#' @examples
get_word_freq_tbl <- function(x, source = "subtlex",
                              language = "english",
                              location = "US",
                          ignore_case = TRUE,
                          remove_punc = "[[:punct:][:blank:]]",
                          regex = FALSE){

  if(source == "subtlex" & language == "english" & location == "US") {
   if(is.null(.pkg_env$subtlexus)) create_SUBTLEXus_rds()
     df_freq <- .pkg_env$subtlexus
  } else {
    stop("Source/language/location is not available.", call. = FALSE)
  }
  if(ignore_case == TRUE) {
    df_freq <- tidytable::mutate.(df_freq, word = tolower(word))
    x <- tolower(x)
  }
  if(remove_punc != FALSE & length(remove_punc) != 0) {
    x <-  chr_remove(x, remove_punc)
  }
  if(regex) {
    data.table::like(vector, pattern, ignore.case = FALSE, fixed = FALSE)
  } else {
    df_freq[data.table::chmatch(x,df_freq$word),]
  }
}

create_SUBTLEXus_rds <- function(force = FALSE){
  fileRDS <- "subtlexus.RDS"
  if(!force){
  file_1 <- file.path(rappdirs::user_data_dir("pangolang"), fileRDS)
  file_2 <- file.path(tempdir(), fileRDS)
  which_file <- which.max(c(file.mtime(file_1), file.mtime(file_2)))
  if(length(which_file)!=0){
    # subtlexus <<- readRDS(c(file_1, file_2)[which_file])
    assign("subtlexus", readRDS(c(file_1, file_2)[which_file]), envir=.pkg_env)

    return(invisible())
  }
  }
  url <- "https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus2.zip"
  file <- file.path(tempdir(),"subtlexus2.zip")
  if(interactive()){
    choices <- c("yes","no")
  choice <- menu(choices, title = paste0("Do you want to download subtlexus data set from [",url,"] ?"))
  if(choices[choice] != "yes") stop("Cannot continue without downloading the dataset."
                      )
  }
  httr::GET(url,
            httr::write_disk(file, overwrite = TRUE),
            httr::progress())
  metadata <- unzip(file, exdir = tempdir(), list = TRUE)
  utils::unzip(file, exdir = tempdir())
  subtlexus <- tidytable::fread.(file.path(tempdir(),metadata$Name)) %>%
    tidytable::rename.(word = Word,
                       freq_count = FREQcount,
                       cd_count = CDcount,
                       freq_count_lc = FREQlow,
                       cd_count_lc = Cdlow,
                       freq_per_million = SUBTLWF,
                       log10_freq_count = Lg10WF,
                       cd_percent = SUBTLCD,
                       log10_cd_count = Lg10WF)

  attributes(subtlexus)$original_filename <- metadata$Name
  attributes(subtlexus)$original_date <- metadata$Date
  attributes(subtlexus)$download_date <- Sys.Date()
  assign("subtlexus", subtlexus, envir=.pkg_env)
  writeRDS(subtlexus,filename = fileRDS)
  invisible()
}

#' Title
#'
#' @param x
#' @param source
#' @param language
#' @param location
#'
#' @return
#' @export
#'
#' @examples
get_all_word_freq_tbl <- function(x, source = "subtlex",
                              language = "english",
                              location = "US"){
  if(source == "subtlex" & language == "english" & location == "US") {
    if(is.null(.pkg_env$subtlexus)) {
      create_SUBTLEXus_rds()
    }
    return(.pkg_env$subtlexus)
  } else {
    stop("Source/language/location is not available.", call. = FALSE)
  }
}
