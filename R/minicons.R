
#' Get the log probability of each word of a text given its previous context.
#'
#' @param x Text
#' @param model Name of the model, should either be a path  to a model (.pt or .bin file) stored locally, or a  pretrained model stored on the Huggingface Model Hub.
#' @param device device type that the model should be loaded on, options: "cpu" or "cuda:{0, 1, ...}"?
#'
#' @return
#'
#' @examples
#' @export
get_text_logprob <- function(x, model = "gpt2", device = "cpu"){
  if(!memoise::has_cache(incremental_LM_scorer)(model, device) ){
    message_verbose("Loading model ", model," in ", device,"...\n")
  }
  LM <- incremental_LM_scorer(model, device)
  py_logprobs <- LM$logprobs(LM$prepare_text(x))
  tidytable::tidytable(logprob = reticulate::py_to_r(py_logprobs)[[1]][[1]]$numpy(),
                       token = reticulate::py_to_r(py_logprobs)[[1]][[2]] %>%
                         chr_remove("Ġ"))
}


#' Get the log probability of each word of a vector of words given its previous context.
#'
#' @param x Vector of words or texts.
#' @param by Vector that indicates how the text should be split
#' @param model Name of the model, should either be a path  to a model (.pt or .bin file) stored locally, or a  pretrained model stored on the Huggingface Model Hub.
#' @param device device type that the model should be loaded on, options: "cpu" or "cuda:{0, 1, ...}"?
#'
#' @return
#'
#' @examples
#' @export
get_word_logprob <- function(x, by = rep(1, length(x)), model = "gpt2", device = "cpu", return_token = TRUE){
  texts <- split(x, by)
  word_logprob_ls <- lapply(texts, function(t){
    text <- t %>% paste(collapse = " ")
    message_verbose("Probabilities for text:\n '", text, "'")
    tokens <- get_text_logprob(text, model = model, device = device)
    out_t <- tidytable::tidytable(x = t, logprob =0,token = "|")
    r <- 1
    for(i in 1:nrow(tokens)){
      #i=2
      out_t$logprob[r] <- out_t[r,]$logprob + tokens$logprob[i]
      out_t$token[r] <- paste0(out_t[r,]$token, tokens$token[i],"|")
      if(out_t$x[r] == tokens$token[i] || chr_ends(out_t$x[r], tokens$token[i])){
        r <- r+1
      }
    }
    out_t
  })

  out <- tidytable::bind_rows.(word_logprob_ls) %>% tidytable::select.(-x)
  if(return_token){
    return(out)
  } else {
    return(tidytable::pull.(out, logprob))
  }
}

#' @noRd
incremental_LM_scorer_ <- function(model, device) {
  minicons$scorer$IncrementalLMScorer(model, device)
}

#' @noRd
incremental_LM_scorer <- memoise::memoise(incremental_LM_scorer_)
