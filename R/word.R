#' Get the word frequency per million words of a string
#'
#' @param x
#' @param source
#' @param ignore_case
#' @param remove_punc
#'
#' @return
#' @export
#'
#' @examples
#'  get_word_fpmw(x = c("car", "cat", "bubble"))
#'
get_word_fpmw <- function(x, source = "SUBTLEX_US",
                          ignore_case = TRUE,
                          remove_punc = "[[:punct:][:blank:]]"){
  df_source <- get_source(source, measure="fpmw", ignore_case = ignore_case)
  x <- clean_string(x, ignore_case, remove_punc)
  df_source[data.table::chmatch(x,df_source$string),] %>%
    tidytable::pull.(name = "string")
}


#' Get the word frequency per million words using regex
#'
#' @param x
#' @param source
#' @param ignore_case
#' @param remove_punc
#'
#' @return
#' @export
#'
#' @examples
#'  get_regex_fpmw_lst(x = c("car", "cat", "bubble"))
#'
get_regex_fpmw_lst <- function(x, source = "SUBTLEX_US",
                               ignore_case = TRUE){

  list_sel <- lapply(x, function(w) {
    out <- df_freq[data.table::like(df_freq$string, pattern = w, ignore.case = FALSE, fixed = FALSE),] %>%
      tidytable::select.(string, tidyselect::all_of(fpmw_source))
    out[complete.cases(out)]
  })
  names(list_sel) <- x
  list_sel
}

#' @noRd
get_source <- function(source, measure="fpmw", ignore_case = ignore_case){
  rmeasure <- paste0("^",measure,"\\.")
  sources <- colnames(LexOPS::lexops) %>% .[chr_detect(.,rmeasure)]
  valid_source <- sub(rmeasure, "", sources)
  if (!source %in% valid_source) {
    stop2("Source '", source, "' is not a valid ",measure," source. ",
          "Valid sources are:\n", collapse_comma(valid_source))
  }
  m_source <- paste0(measure,".", source)
  df <- tidytable::as_tidytable(LexOPS::lexops)
  tidytable::mutate.(df, string = tolower(string)) %>%
    tidytable::select.(string, tidyselect::all_of(m_source))
}

#' @noRd
clean_string <- function(x, ignore_case, remove_punc){
  if(ignore_case == TRUE) {
    x <- tolower(x)
  }
  if(remove_punc != FALSE & length(remove_punc) != 0) {
    x <-  chr_remove(x, remove_punc)
  }
  x
}


