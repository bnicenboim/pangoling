
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
get_text_log_prob <- function(x, model = "gpt2", device = "cpu"){
  if(!memoise::has_cache(incremental_LM_scorer)(model, device) ){
    message_verbose("Loading model ", model," in ", device,"...\n")
  }
  LM <- incremental_LM_scorer(model, device)
  py_log_probs <- LM$logprobs(LM$prepare_text(x))
  tidytable::tidytable(log_prob = reticulate::py_to_r(py_log_probs)[[1]][[1]]$numpy(),
                       token = reticulate::py_to_r(py_log_probs)[[1]][[2]] %>%
                         chr_remove("Ġ"))
}


#' Get the log probability of each word of a vector of words given its previous context.
#'
#' @param x Vector of words, phrases or texts.
#' @param by Vector that indicates how the text should be split
#' @param model Name of the model, should either be a path  to a model (.pt or .bin file) stored locally, or a  pretrained model stored on the Huggingface Model Hub.
#' @param device device type that the model should be loaded on, options: "cpu" or "cuda:{0, 1, ...}"?
#'
#' @return
#'
#' @examples
#' @export
get_word_log_prob <- function(x, by = rep(1, length(x)), model = "gpt2", device = "cpu"){
  texts <- split(x, by)
  N <- length(texts)
  word_log_prob_ls <- tidytable::map2.(texts, seq_along(texts), function(t,i){
    text <- t %>% paste(collapse = " ")
    message_verbose("Probabilities for text (",i,"/",N,"):\n '", text, "'")
    tokens <- get_text_log_prob(text, model = model, device = device)
    out_t <- tidytable::tidytable(x = t, log_prob =0,token = "|")
    r <- 1
    for(i in 1:nrow(tokens)){
      #i=2
      out_t$log_prob[r] <- out_t[r,]$log_prob + tokens$log_prob[i]
      out_t$token[r] <- paste0(out_t[r,]$token, tokens$token[i],"|")
      if(out_t$x[r] == tokens$token[i] || chr_ends(out_t$x[r], tokens$token[i])){
        r <- r+1
      }
    }
    out_t
  })

  tidytable::bind_rows.(word_log_prob_ls) %>% tidytable::rename.(phrase = x)
}

#' @noRd
incremental_LM_scorer_ <- function(model, device) {
  minicons$scorer$IncrementalLMScorer(model, device)
}

#' @noRd
incremental_LM_scorer <- memoise::memoise(incremental_LM_scorer_)
