#' Get the log probability of each word phrase of a vector given its previous context using a transformer model from huggingface.co/.
#'
#' In case of errors check the status of https://status.huggingface.co/
#'
#' @param x Vector of words, phrases or texts.
#' @param by Vector that indicates how the text should be split.
#' @param model Name of a pretrained model stored on the huggingface.co. (Maybe a path to a  model (.pt or .bin file) stored locally will work.)
#' @ignore_regex Can ignore certain characters when calculates the log probabilities. For example `^[[:punct:]]$` will ignore all punctuation  that stands alone in a token.
#' @eot 0 Does nothing, 1 adds an end of text tag at the beginning of a vector, 2 adds an end of text tag and a space at the beginning of a vector.
#'
#' @return a vector of log probabilities.
#'
#' @examples
#' @export
get_causal_log_prob <- function(x, by = rep(1, length(x)), ignore_regex = "",  model = "gpt2",eot = 0) {
  get_lm_lp(x =x, by=by, ignore_regex = ignore_regex, type = "causal",model=model, eot =eot, npred = 1)
}


#' Title
#'
#' Incorrect for multi-token words.
#'
#' @param x
#' @param by
#' @param eot
#' @param model
#'
#' @return
#' @export
#'
#' @examples
get_causal_entropy <- function(x, by = rep(1, length(x)), model = "gpt2", eot = 0) {
  x <- trimws(x, whitespace = "[ \t]")
  texts <- split(x, by)
  N <- length(texts)
  out <- tidytable::map2.(texts,names(texts), function(words,item) {
    # words <- texts[[1]]
    mat <- causal_log_prob_mat(words, eot = eot)
    #vocab <- get_tr_vocab(model)
    if(length(words) >1){
      words_lm <- c(words[1], paste0(" ", words[-1]))
    } else {
      words_lm <- words
    }
    tokens <- get_token.list(get_id(words_lm, model), model)
    token_n <- tidytable::map_dbl.(tokens, length)
    #index_vocab <- data.table::chmatch(unlist(tokens), vocab)
    token_entropy <- apply(mat, 2, function(lp) -sum(exp(lp)*lp))
    message_verbose("Text id: ",item,"\n`", text,"`")
    if(options()$pangolang.debug) {
      print("******")
      sent <- tidytable::map_chr.(tokens, function(x) paste0(x, collapse = "|"))
      print(paste0("[",sent,"]", collapse = " "))
      print(token_entropy)
    }
    n <- 1
    word_entropy <- vector(mode="numeric", length(words))
    for(i in seq_along(token_n)){
      t <- token_n[i]
      if(eot!=0 && n ==1){
        # ignores the NA in the first column
        word_entropy[i] <- sum(token_entropy[(n+1):(n+(t-1))])
      } else {
        word_entropy[i] <- sum(token_entropy[n:(n+(t-1))])
      }
      n <- n + t
    }
    word_entropy
  })
  out <- unlist(out)
  names(out) <- x
  out
}
#' Title
#'
#' @param x
#' @param by
#' @param eot
#' @param model
#'
#' @return
#' @export
#'
#' @examples
get_causal_log_prob_mat <- function(x, by = rep(1, length(x)),  model = "gpt2", eot = 0) {
  x <- trimws(x, whitespace = "[ \t]")
  texts <- split(x, by)
  N <- length(texts)
  tidytable::map2.(texts,names(texts), function(words,item) {
    causal_log_prob_mat(words, eot= eot, model = model)

  })
}

causal_log_prob_mat <- function(words, model = "gpt2", eot = 0){
  if(eot !=0) {
    eos <- tokenizer(model)$eos_token
    eos_t <- tokenizer(model)$convert_tokens_to_string(eos)
    if(eot==1)words[1] <- paste0(eos_t,words[1])
    if(eot==2) words[1] <- paste0(eos_t," ",words[1])
  }
  text <-  paste0(words, collapse = " ")
  tensor <- tokenizer(model)(text, return_tensors = "pt")$input_ids
  out_lm <- lang_model(model, task = "causal")(tensor)
  logits <- out_lm$logits[0]
  tokens <- reticulate::py_to_r(tokenizer(model)$convert_ids_to_tokens(tensor[0]))
  lp <- reticulate::py_to_r(torch$log_softmax(logits, dim=-1L))$tolist()
  mat <- do.call("cbind",lp)
  # remove the last prediction, and the first is NA
  mat <- cbind(rep(NA,nrow(mat)), mat[,-ncol(mat)])
  rownames(mat)  <- get_tr_vocab(model)
  colnames(mat) <- unlist(tokens)
  list(mat)
}



#' Get the log probability of each word phrase of a vector given its previous context using a transformer model from huggingface.co/.
#'
#' @param context Context
#' @param model Name of a pretrained model stored on the huggingface.co. (Maybe a path to a  model (.pt or .bin file) stored locally will work.)
#'
#' @return
#'
#' @examples
#' @export
#' @export
get_causal_next_tokens_tbl <- function(context, model = "gpt2") {
  context_tensor <-
    tokenizer(model)(context, return_tensors = "pt")$input_ids

  generated_outputs <- lang_model(model, "causal")(context_tensor)
  n_tokens <- length(context_tensor$tolist()[0])
  logits_next_word <- generated_outputs$logits[0][n_tokens-1]
  lp <- reticulate::py_to_r(torch$log_softmax(logits_next_word, dim = -1L)$tolist())%>% unlist()

  tidytable::tidytable(token = get_tr_vocab(model),  log_prob = lp) %>%
    tidytable::arrange.(-log_prob)
}

