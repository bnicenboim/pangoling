#' Get a table with word frequencies.
#'
#' @param x
#' @param source
#' @param ignore_case
#' @param remove_punc
#' @param regex
#'
#' @return
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
#'
#' The dominant (most frequent) PoS of each entry
#' The frequency of the dominant PoS
#' The relative frequency of the dominant PoS
#' All PoS observed for the entry
#' The frequencies of each PoS
#' @export
#'
#' @examples
get_word_freq_tbl <- function(x, source =list(language = "english",
                                              location = "US"),
                          ignore_case = TRUE,
                          remove_punc = "[[:punct:][:blank:]]",
                          regex = FALSE){

  df_freq <- get_all_word_freq_tbl(source)

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

freq_source <- list(subtlexus = list(
  description = "SUBTLEX-US word frequencies for American English with parts of speech",
  url = "http://crr.ugent.be/papers/SUBTLEX-US_frequency_list_with_PoS_information_final_text_version.zip",
  name = "subtlexus-pos",
  language = "english",
  location = "US",
  filetype = "zip",
  doi = c("10.3758/BRM.41.4.977", "10.3758/s13428-012-0190-4")),
  subtlexnl = list(
    description = "SUBTLEX-NL word frequencies for Dutch with parts of speech",
    url = "http://crr.ugent.be/subtlex-nl/SUBTLEX-NL.cd-above2.with-pos.txt.zip",
    name = "subtlexnl-pos",
    language = "dutch",
    location = "NL",
    filetype = "zip",
    doi = c("10.3758/BRM.42.3.643"))
)


create_rds <- function(source, force = FALSE){
  fileRDS <- paste0(source$name,".RDS")
  if(!force){
  file_1 <- file.path(rappdirs::user_data_dir("pangolang"), fileRDS)
  file_2 <- file.path(tempdir(), fileRDS)
  which_file <- which.max(c(file.mtime(file_1), file.mtime(file_2)))
  if(length(which_file)!=0){
    # subtlexus <<- readRDS(c(file_1, file_2)[which_file])
    assign(source$name, readRDS(c(file_1, file_2)[which_file]), envir=.pkg_env)
    return(invisible())
    }
  }
  file <- file.path(tempdir(),paste0(source$name,".",source$filetype))
  if(interactive()){
    choices <- c("yes","no")
  choice <- menu(choices, title = paste0("Do you want to download ", source$name," data set from [",source$url,"] ?"))
  if(choices[choice] != "yes") stop("Cannot continue without downloading the dataset."
                      )
  }
  httr::GET(source$url,
            httr::write_disk(file, overwrite = TRUE),
            httr::progress())
  if(source$filetype =="zip"){
    metadata <- unzip(file, exdir = tempdir(), list = TRUE)
    utils::unzip(file, exdir = tempdir())
    file <- file.path(tempdir(),metadata$Name)
  } else if(source$filetype =="xlsx"){
  #TODO
    }

  namekey <- c(freq_count = "FREQcount",
          cd_count = "CDcount",
    freq_count_lc = "FREQlow",
    cd_count_lc = "Cdlow",
    cd_count_lc = "CDlow",
    freq_per_million = "SUBTLWF",
    freq_per_million = "SUBTLEXWF",
    freq_lemma = "FREQlemma",
    log10freq_count = "Lg10WF",
    cd_percent = "SUBTLCD",
    cd_percent = "SUBTLEXCD",

    log10cd_count = "Lg10WF",
    log10_cd = "Lg10CD",
    dom_pos = "Dom_PoS_SUBTLEX",
    dom_pos = "dominant.pos",

    freq_dom_pos = "Freq_dom_PoS_SUBTLEX",
    freq_dom_pos = "dominant.pos.freq",
    dom_pos_perc = "Percentage_dom_PoS",
    dom_pos_lemma = "dominant.pos.lemma",
    all_pos = "All_PoS_SUBTLEX",
    all_pos = "all.pos",
    freq_all_pos = "All_freqs_SUBTLEX",
    freq_all_pos = "all.pos.freq",
    freq_all_pos_lemma = "all.pos.lemma.freq",
    word = "Word"
    )

  freq_tbl <- tidytable::fread.(file)
  old_names <- names(freq_tbl)
  new_names <- names(namekey)
  names(freq_tbl)[old_names %in% namekey] <- names(namekey[match(old_names, namekey, nomatch= 0) ])

  # attributes(freq_tbl)$original_filename <- metadata$Name
  # attributes(freq_tbl)$original_date <- metadata$Date
  attributes(freq_tbl)$download_date <- Sys.Date()
  assign(source$name, freq_tbl, envir=.pkg_env)
  writeRDS(freq_tbl,filename = fileRDS)
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
get_all_word_freq_tbl <- function(source =list(
                                                  language = "english")){
  possible_sources <- tidytable::map2.(source, names(source), ~ which(sapply(freq_source, function(x) x[[.y]] == .x)))
  source_index <- Reduce(intersect, possible_sources)
  if(length(source_index)==0) stop("Source/language/location is not available.", call. = FALSE)
  source <- freq_source[[source_index]]

  if(is.null(.pkg_env[[source$name]])) create_rds(source)

 .pkg_env[[source$name]]
}
