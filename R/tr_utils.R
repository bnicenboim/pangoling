#' Sends a var to python
#' https://stackoverflow.com/questions/67562889/interoperability-between-python-and-r
var_to_py <- function(var_name, x ) {
e <- new.env()
options("reticulate.engine.environment" = e)
assign(var_name, x, envir = e)
#options("reticulate.engine.environment" = NULL)
}

lst_to_kwargs <- function(x){
 x <- x[lengths(x) > 0]
 if(!is.list(x)) x <- as.list(x)
 x <- reticulate::r_to_py(x)
 var_to_py("kwargs", x)
}

#' @noRd
lm_init <- function(model = "gpt2", task = "causal", config = list()) {
  reticulate::py_run_string('import os\nos.environ["TOKENIZERS_PARALLELISM"] = "false"')

  # to prevent memory leaks:
  reticulate::py_run_string('there = "lm" in locals()')
  if(reticulate::py$there) reticulate::py_run_string("del lm")
  reticulate::py_run_string("import torch
torch.cuda.empty_cache()")
  gc(full = TRUE)
  reticulate::py_run_string("import gc
gc.collect()")
  gc(full = TRUE)

  # disable grad to speed up things
  torch$set_grad_enabled(FALSE)

  reticulate::py_run_string("import transformers")
  automodel <- switch(task,
         causal= "AutoModelForCausalLM",
         masked = "AutoModelForMaskedLM"
         )
  # dots <- list(...)
  # args <- c(pretrained_model_name_or_path = model, return_dict_in_generate =TRUE, dots)
  # args <- args[lengths(args) > 0] # remove empty elements
  # extra_args <- reticulate::r_to_py(args)
  # var_to_py("extra_args", extra_args)
  lst_to_kwargs(c(pretrained_model_name_or_path = model, return_dict_in_generate = TRUE, config))
  reticulate::py_run_string(paste0("lm = transformers.",automodel,".from_pretrained(**r.kwargs)"))

#   remove:
#   if(length(unlist(dots))>0) {
#
#     reticulate::py_run_string(paste0("lm = transformers.",automodel,".from_pretrained(r.model, return_dict_in_generate =True, **r.extra_args)"))
#   } else {
#     reticulate::py_run_string(paste0("lm = transformers.",automodel,".from_pretrained(r.model, return_dict_in_generate =True)"))
# }
  lm <- reticulate::py$lm
  lm$eval()

  options("reticulate.engine.environment" = NULL)  # unset option
  # trys to remove everything from memory
  reticulate::py_run_string("import gc
gc.collect()")
  gc(full = TRUE)

  lm
}

#' https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoTokenizer
#' @noRd
tokenizer_init <- function(model = "gpt2", add_bos_token = NULL, config = NULL) {
  reticulate::py_run_string("import transformers")
  if(model == "gpt2" && !is.null(add_bos_token)){
    lst_to_kwargs(c(pretrained_model_name_or_path = model, add_bos_token = add_bos_token, config))
    reticulate::py_to_r(reticulate::py_run_string("tkzr = transformers.GPT2Tokenizer.from_pretrained(**r.kwargs)"))
    #
    # tkzr <- transformers$GPT2Tokenizer$from_pretrained(model,add_bos_token = add_bos_token, ...)
  } else {
    lst_to_kwargs(c(pretrained_model_name_or_path = model, config))

    ## extra_args <- reticulate::r_to_py(args)
    ## var_to_py("extra_args", extra_args)
    reticulate::py_to_r(reticulate::py_run_string("tkzr = transformers.AutoTokenizer.from_pretrained(**r.kwargs)"))
    # tkzr <- transformers$AutoTokenizer$from_pretrained(model, ...)
  }

  tkzr <- reticulate::py$tkzr

  # trys to remove everything from memory
  reticulate::py_run_string("import gc
gc.collect()")
  gc(full = TRUE)
  tkzr
}

#' @noRd
tokenizer <- memoise::memoise(tokenizer_init)

#' @noRd
lang_model <- memoise::memoise(lm_init)



#' @noRd
get_vocab_init <- function(model = "gpt2", add_bos_token = NULL, config = list()) {
  tkzr <- tokenizer(model, add_bos_token = add_bos_token, config = config)
  size <- tkzr$vocab_size
  tkzr$convert_ids_to_tokens(0L:(size -
                                                                 1L))
}

#' Returns the vocabulary of a model
#'
#' @inheritParams get_causal_log_prob
#'
#' @return A vector with the vocabulary of a model
#' @export
#'
#'
get_tr_vocab <- memoise::memoise(get_vocab_init)

#' Get ids (without adding special characters at beginning or end?)
get_id <- function(x, model = "gpt2", add_bos_token = NULL, config = NULL){
  lapply(x, function(i){
    t <- tokenizer(model, add_bos_token = add_bos_token, config = config)$tokenize(i)
    tokenizer(model, add_bos_token = add_bos_token, config = config)$convert_tokens_to_ids(t)
  } )
}


#' Tokenize the input
#'
#' @param x input
#' @inheritParams get_causal_log_prob
#'
#' @return
#' @export
#'
#' @examples
get_tokens <- function(x, model = "gpt2", add_bos_token = NULL, config = NULL) {
  UseMethod("get_tokens")
}

#' @export
get_tokens.character <- function(x, model = "gpt2", add_bos_token = NULL, config = NULL) {
  id <- get_id(x, model = model, add_bos_token= add_bos_token, config = config)
  lapply(id, function(i) get_tokens.numeric(i, model = model))
}

#' @export
get_tokens.numeric <- function(x, model = "gpt2", add_bos_token = NULL, config = NULL){
  tidytable::map_chr.(as.integer(x), function(x)
    tokenizer(model, add_bos_token = add_bos_token, config = config)$convert_ids_to_tokens(x))
}

#' The number of tokens in a string or vector of strings
#'
#' @param x character input
#' @inheritParams get_causal_log_prob
#'
#' @return The number of tokens in a string or vector of words.
#'
#' @export
#'
#' @examples
ntokens <- function(x, model = "gpt2", add_bos_token = NULL, config = NULL) {
  length(unlist(get_tokens(x, model, add_bos_token = add_bos_token, config = config), recursive = TRUE))
}

#' @noRd
get_lm_lp <- function(x, by= rep(1, length(x)), ignore_regex = "", type = "causal", model = "gpt2", n_plus = 3,...) {
  if(length(x)!= length(by)) stop2("The argument `by` has an incorrect length.")
  if(length(x) <=1 ) stop2("The argument `x` needs at least two elements.")
  x <- trimws(x, whitespace = "[ \t]")
  texts <- split(x, by)
  N <- length(texts)
  out <- tidytable::map2.(texts,names(texts), function(words,item) {
    # words <- texts[[1]]
    # item <- names(texts[1])
    if(type == "causal")  ls_mat <- causal_log_prob_mat(words, model = model, ...)
    if(type == "masked")  ls_mat <- masked_log_prob_mat(words, model = model, n_plus =n_plus, ...)


    if(length(words) >1){
      words_lm <- c(words[1], paste0(" ", words[-1]))
    } else {
      words_lm <- words
    }

    tokens <- lapply(get_id(words_lm, model), get_tokens.numeric, model = model)
    token_n <- tidytable::map_dbl.(tokens, length)
    index_vocab <- data.table::chmatch(unlist(tokens), vocab)
    message_verbose("Text id: ",item,"\n`", paste(words, collapse =" "),"`")
    # if(length(ls_mat)>1){
    # # when there is a matrix for the predictions made from each word,
    # # remove the predictions made in a token in the middle of a word:
    # # for example from 'nt
    # token_remove <- tidytable::map.(tokens, ~ cumsum(seq_along(.x)) >1) |>
    #   unlist()
    # ls_mat[token_remove] <- NULL
    # }

    # predictions for word n+1, n+2, etc...
    if(n_plus==0) n_plus <- length(ls_mat)
    ls_preds <-  lapply(1:n_plus, function(p){
      # when p = 1, predictions for token[i] are made in token[i-1]
      mat <- ls_mat[[p]]
      #mat[vocab =="isn",2]
      token_lp <- tidytable::map2_dbl.(index_vocab,1:ncol(mat), ~ mat[.x,.y])

      if(options()$pangoling.debug) {
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
      # i <- 1
      for(i in  seq_along(token_n)){
        t <- token_n[i]
        n_p <- n-(p-1) #source of the prediction
        if(n_p <1 || !n_p %in% c(cumsum(c(0,token_n))+1)) {
          word_lp[i] <- NA
        } else {
          word_lp[i] <- sum(token_lp[n:(n+(t-1))])
        }
        n <- n + t
        # i <- i + 1
      }
      word_lp
    })
    pred_mat <- do.call("rbind",ls_preds)
    #change order so that they start from pred n+1
    #pred_mat <- pred_mat[nrow(pred_mat):1, , drop = FALSE]
    colnames(pred_mat)  <- words_lm
    lapply(as.list(as.data.frame(pred_mat)), function(x){
      #removes the NA from the front
      x <- x[!is.na(x)]
      # puts them at the end
      length(x) <- n_plus
      x
      })
  })
 out <- unlist(out, recursive = FALSE)
 # names(out) <- x
  out
}


#' @noRd
slide_tokens <- function(input_ids,max_tokens, stride ){
message_verbose("Number of tokens larger than the maximum allowed ",max_tokens,". Using a sliding window." )
#build a matrix with max tokens rows, and as many columns as needed
ids_matrix <- embed(input_ids,max_tokens)[,max_tokens:1]
rel_rows <- c(seq(1,nrow(ids_matrix)-1, stride),nrow(ids_matrix))
rel_token_pos <-  diff(rel_rows)
ids_matrix <- ids_matrix[rel_rows,]
lapply(seq_len(nrow(ids_matrix)), function(i) ids_matrix[i, ])
}

#' @noRd
rel_pos_slide <- function(input_ids,max_tokens, stride ){
  ids_matrix <- embed(input_ids,max_tokens)[,max_tokens:1]
  rel_rows <- c(seq(1,nrow(ids_matrix)-1, stride),nrow(ids_matrix))
  c(list(1:max_tokens), lapply(diff(rel_rows), function(.x) seq.int(from = max_tokens-.x+1,to = max_tokens)))
}
