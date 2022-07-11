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
                          remove_punc = "'s|[[:punct:][:blank:]]$",
                          nomatch = NA){
  df_source <- get_source(source, feature="fpmw", ignore_case = ignore_case)
  x <- clean_string(x, ignore_case, remove_punc)
  df_source[data.table::chmatch(x,df_source$string, nomatch = nomatch),] |>
    tidytable::pull.(name = "string")
}



#' Get the word feature
#'
#' @param x
#' @param source
#' @param ignore_case
#' @param remove_punc
#'
#' @return
#'
#' @examples
#'  get_word_feature(x = c("car", "cat", "bubble"))
#' @export
get_word_feature <- function(x, source = "SUBTLEX_US",
                          ignore_case = TRUE,
                          feature = "Zipf",
                          remove_punc = "[[:punct:][:blank:]]"){
  df_source <- get_source(source, feature=feature, ignore_case = ignore_case)
  x <- clean_string(x, ignore_case, remove_punc)
  df_source[data.table::chmatch(x,df_source$string),] |>
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
    out <- df_freq[data.table::like(df_freq$string, pattern = w, ignore.case = FALSE, fixed = FALSE),] |>
      tidytable::select.(string, tidyselect::all_of(fpmw_source))
    out[complete.cases(out)]
  })
  names(list_sel) <- x
  list_sel
}

#' @noRd
get_source <- function(source, feature="fpmw", ignore_case = ignore_case){
  feat_source <- colnames(LexOPS::lexops)
  m_source <- tolower(paste0(feature,".", source))
  rfeature <- tolower(paste0("^",feature,"\\."))
  valid_source <- chr_replace_all(feat_source[-1],"^.*?\\.", "")
  valid_features <- chr_extract(feat_source[-1], "^.*?\\.") |>
    chr_remove(".$")
  col <- feat_source[tolower(feat_source)==tolower(m_source)]
  if(length(col)=="") {
    stop2("Feature '", feature, "' is not a valid feature for source ", source,". Valid source - features combinations are:\n", collapse_comma(paste0("'",valid_source,"'-'", valid_features,"'")))
  }
  #|>
  #   .[chr_detect(.,rfeature)]
  #
  tidytable::as_tidytable(LexOPS::lexops) |>
  tidytable::mutate.(string = tolower(string)) |>
    tidytable::rename_with.(tolower) |>
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


