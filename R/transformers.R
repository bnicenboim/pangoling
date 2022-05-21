#' Get the log probability of each word phrase of a vector given its previous context using a transformer model from huggingface.co/.
#'
#' In case of errors check the status of https://status.huggingface.co/
#'
#' @param x Vector of words, phrases or texts.
#' @param by Vector that indicates how the text should be split.
#' @param model Name of a pretrained model stored on the huggingface.co. (Maybe a path to a  model (.pt or .bin file) stored locally will work.)
#' @eot 0 Does nothing, 1 adds an end of text tag at the beginning of a vector, 2 adds an end of text tag and a space at the beginning of a vector.
#'
#' @return a vector of log probabilities.
#'
#' @examples
#' @export
get_tr_log_prob <- function(x, by = rep(1, length(x)),eot = 0, model = "gpt2") {
  x <- trimws(x, whitespace = "[ \t]")
  texts <- split(x, by)
  N <- length(texts)
  out <- tidytable::map2.(texts,names(texts), function(words,item) {
    # words <- texts[[1]]
    mat <- tr_log_prob_mat(words, eot = eot)
    vocab <- get_vocab(model)
    if(length(words) >1){
      words_lm <- c(words[1], paste0(" ", words[-1]))
    } else {
      words_lm <- words
    }
    tokens <- get_token.list(get_id(words_lm))
    token_n <- tidytable::map_dbl.(tokens, length)
    index_vocab <- data.table::chmatch(unlist(tokens), vocab)
    token_lp <- tidytable::map2_dbl.(index_vocab,1:ncol(mat), ~ mat[.x,.y])
    message_verbose("Text id: ",item,"\n`", text,"`")
    if(options()$pangolang.debug) {
      print("******")
      sent <- tidytable::map_chr.(tokens, function(x) paste0(x, collapse = "|"))
      print(paste0("[",sent,"]", collapse = " "))
      print(token_lp)
    }

    n <- 1
    word_lp <- vector(mode="numeric", length(words))
    for(i in seq_along(token_n)){
      t <- token_n[i]
      if(eot!=0 && n ==1){
        # ignores the NA in the first column
        word_lp[i] <- sum(token_lp[(n+1):(n+(t-1))])
      } else {
        word_lp[i] <- sum(token_lp[n:(n+(t-1))])
      }
      n <- n + t
    }
    word_lp
  })
  out <- unlist(out)
  names(out) <- x
  out
}

get_tr_log_prob_mat <- function(x, by = rep(1, length(x)),eot = 0, model = "gpt2") {
  x <- trimws(x, whitespace = "[ \t]")
  texts <- split(x, by)
  N <- length(texts)
  out <- tidytable::map2.(texts,names(texts), function(words,item) {
    tr_log_prob_mat(words, eot= eot)

  })
}

tr_log_prob_mat <- function(words, eot = 0){
  if(eot==1){
    words[1] <- paste0("<|endoftext|>",words[1])
  } else if(eot==2){
    words[1] <- paste0("<|endoftext|> ",words[1])
  }
  text <-  paste0(words, collapse = " ")
  tensor <- tokenizer(model)(text, return_tensors = "pt")$input_ids
  out_lm <- lang_model(model)(tensor)
  logits <- out_lm$logits[0]
  tokens <- reticulate::py_to_r(tokenizer(model)$convert_ids_to_tokens(tensor[0]))
  lp <- reticulate::py_to_r(torch$log_softmax(logits, dim=-1L))$tolist()
  mat <- do.call("cbind",lp)
  # remove the last prediction, and the first is NA
  mat <- cbind(rep(NA,nrow(mat)), mat[,-ncol(mat)])
  rownames(mat)  <- get_vocab(model)
  colnames(mat) <- unlist(tokens)
  mat
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
get_tr_next_tokens_tbl <- function(context, model = "gpt2") {
  # lang_model(model)$eval()
  context_tensor <-
    tokenizer(model)(context, return_tensors = "pt")$input_ids

  generated_outputs <- lang_model(model)(context_tensor)
  n_tokens <- length(context_tensor$tolist()[0])
  logits_next_word <- generated_outputs$logits[0][n_tokens-1]
  lp <- reticulate::py_to_r(torch$log_softmax(logits_next_word, dim = -1L)$tolist())%>% unlist()

  tidytable::tidytable(token = get_vocab(model),  log_prob = lp) %>%
    tidytable::arrange.(-log_prob)
}

#' @noRd
lang_model_init <- function(model = "gpt2") {
  torch$set_grad_enabled(FALSE)
  lm <- reticulate::py_to_r(transformers$AutoModelForCausalLM$from_pretrained(model, return_dict_in_generate =TRUE))
  lm$eval()
  lm
}
#' @noRd
tokenizer_init <- function(model = "gpt2") {
  reticulate::py_to_r(transformers$AutoTokenizer$from_pretrained(model))
}

#' @noRd
tokenizer <- memoise::memoise(tokenizer_init)
#' @noRd
lang_model <- memoise::memoise(lang_model_init)
#' @noRd
get_vocab_init <- function(model = "gpt2") {
  size <- reticulate::py_to_r(tokenizer(model)$vocab_size)
  reticulate::py_to_r(tokenizer(model)$convert_ids_to_tokens(0L:(size -
                                                                   1L)))
}

#' @noRd
get_vocab <- memoise::memoise(get_vocab_init)




get_id <- function(x, model = "gpt2"){
  lapply(x, function(x) reticulate::py_to_r(tokenizer(model)(x, return_tensors = "pt")$input_ids[0]$tolist()))
}

get_token <- function(x, model = "gpt2") {
  UseMethod("get_token")
}

get_token.list <- function(x, model = "gpt2") {
  lapply(x, get_token.numeric)
}
get_token.character <- function(x, model = "gpt2") {
  id <- get_id(x, model)
  get_token.int(id)
}

get_token.numeric <- function(x, model = "gpt2"){
  tidytable::map_chr.(as.integer(x), function(x)
    reticulate::py_to_r(tokenizer(model)$convert_ids_to_tokens(x)))
}
