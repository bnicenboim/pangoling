#' @noRd
lm_init <- function(model = "gpt2", task = "causal") {
  torch$set_grad_enabled(FALSE)
  if(task == "causal")  lm <- reticulate::py_to_r(transformers$AutoModelForCausalLM$from_pretrained(model, return_dict_in_generate =TRUE))
  if(task == "masked") lm <- reticulate::py_to_r(transformers$AutoModelForMaskedLM$from_pretrained(model, return_dict_in_generate =TRUE))
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
lang_model <- memoise::memoise(lm_init)

#' Title
#'
#' @param method
#' @param conda
#'
#' @return
#' @export
#'
#' @examples
install_transformers <- function(method = "auto", conda = "auto") {
  reticulate::py_install("transformers", method = method, conda = conda)
}


#' @noRd
get_vocab_init <- function(model = "gpt2") {
  size <- reticulate::py_to_r(tokenizer(model)$vocab_size)
  reticulate::py_to_r(tokenizer(model)$convert_ids_to_tokens(0L:(size -
                                                                   1L)))
}

#' Title
#'
#' @param model
#'
#' @return
#' @export
#'
#' @examples
get_tr_vocab <- memoise::memoise(get_vocab_init)

#' Get ids without adding special characters at beginning or end
get_id <- function(x, model = "gpt2"){
  lapply(x, function(i){
    t <- tokenizer(model)$tokenize(i)
    reticulate::py_to_r(tokenizer(model)$convert_tokens_to_ids(t))
  } )
}

get_token <- function(x, model = "gpt2") {
  UseMethod("get_token")
}

get_token.list <- function(x, model = "gpt2") {
  lapply(x, get_token.numeric, model = model)
}
get_token.character <- function(x, model = "gpt2") {
  id <- get_id(x, model = model)
  get_token.int(id, model = model)
}

get_token.numeric <- function(x, model = "gpt2"){
  tidytable::map_chr.(as.integer(x), function(x)
    reticulate::py_to_r(tokenizer(model)$convert_ids_to_tokens(x)))
}

####
get_lm_lp <- function(x, by= rep(1, length(x)), ignore_regex = "", type = "causal", model = "gpt2",npred =0,...) {
  x <- trimws(x, whitespace = "[ \t]")
  texts <- split(x, by)
  N <- length(texts)
  out <- tidytable::map2.(texts,names(texts), function(words,item) {
    # words <- texts[[1]]
    #item <- names(texts[1])
    if(type == "causal")  ls_mat <- causal_log_prob_mat(words, model = model, ...)
    if(type == "masked")  ls_mat <- masked_log_prob_mat(words, model = model, ...)
    vocab <- get_tr_vocab(model)
    if(length(words) >1){
      words_lm <- c(words[1], paste0(" ", words[-1]))
    } else {
      words_lm <- words
    }
    tokens <- get_token.list(get_id(words_lm, model), model)
    token_n <- tidytable::map_dbl.(tokens, length)
    index_vocab <- data.table::chmatch(unlist(tokens), vocab)
    message_verbose("Text id: ",item,"\n`", paste(words, collapse =" "),"`")
    # predictions for word n+1, n+2, etc...
    if(npred==0) npred <- length(ls_mat)
    ls_preds <-  lapply(ls_mat[1:npred], function(mat){
      token_lp <- tidytable::map2_dbl.(index_vocab,1:ncol(mat), ~ mat[.x,.y])

      if(options()$pangolang.debug) {
        print("******")
        sent <- tidytable::map_chr.(tokens, function(x) paste0(x, collapse = "|"))
        print(paste0("[",sent,"]", collapse = " "))
        print(token_lp)
      }
      n <- 1
      word_lp <- vector(mode="numeric", length(words))
      # It might ignore punctuation if ignore_regex is used
      if(length(ignore_regex) >0 && ignore_regex != "") {
        pos <- which(grepl(pattern = ignore_regex, x = unlist(tokens)))
        token_lp[pos] <- 0
      }
      # ignores the NA in the first column if it starts with a special character
      if(unlist(tokens)[1] %in% reticulate::py_to_r(tokenizer(model)$all_special_tokens)) token_lp[1] <- 0
      for(i in  seq_along(token_n)){
        t <- token_n[i]
        word_lp[i] <- sum(token_lp[n:(n+(t-1))])
        n <- n + t
      }
      word_lp
    })
    pred_mat <- do.call("rbind",ls_preds)
    colnames(pred_mat)  <- words_lm
    as.list(as.data.frame(pred_mat))
  })
 out <- unlist(out, recursive = FALSE)
 # names(out) <- x
  out
}
