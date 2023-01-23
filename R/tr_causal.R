#' Preloads a causal language model
#'
#' Preloads a causal language model to speed up next runs.
#'
#' @inheritParams get_causal_next_tokens_tbl
#'
#' @return Nothing
#' @export
#'
#' @examples
preload_causal <- function(model = "gpt2", add_bos_token = NULL, config_model = list(), config_tokenizer =list()) {
  lang_model(model, task = "causal", config_model)
  tokenizer(model, add_bos_token = add_bos_token, config_tokenizer)
  invisible()
}

#' Get the log probability of each word phrase of a vector given its previous context.
#'
#' Get the log probability of each word phrase of a vector given its previous context using a causal transformer model from huggingface.com.
#'
#' `add_bos_token` by default  acts as the [AutoTokenizer](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoTokenizer). Using `...` it's possible to control how the model from hugging face is accessed, see [from_pretrained](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained) for details.
#'
#' @param context Context
#' @param model Name of a pretrained model stored on the huggingface.co. (Maybe a path to a  model (.pt or .bin file) stored locally will work.)
#' @param add_bos_token Whether to include beginning of text special tokens. See details
#' @param config_model list with other arguments that control how the model from hugging face is accessed.
#' @param config_tokenizer list with arguments that control how the model from hugging face is accessed.
#'
#' @return
#'
#' @export
get_causal_next_tokens_tbl <- function(context, model = "gpt2", add_bos_token = NULL, config_model = list(), config_tokenizer = list()) {

  tkzr <- tokenizer(model, add_bos_token = add_bos_token, config_tokenizer)
  context_tensor <- tkzr$encode(context, return_tensors = "pt")
  generated_outputs <- lang_model(model, "causal", config_model)(context_tensor)
  n_tokens <- length(context_tensor$tolist()[0])
  logits_next_word <- generated_outputs$logits[0][n_tokens - 1]
  lp <- reticulate::py_to_r(torch$log_softmax(logits_next_word, dim = -1L)$tolist()) |>
    unlist()

  vocab <- get_tr_vocab(model, add_bos_token = add_bos_token, config_tokenizer)

  tidytable::tidytable(token = vocab, log_prob = lp) |>
    tidytable::arrange.(-log_prob)
}


#' Get the log probability of each word phrase of a vector given its previous context using a transformer model from huggingface.co/.
#'
#' In case of errors check the status of https://status.huggingface.co/
#'
#' @param x Vector of words, phrases or texts.
#' @param by Vector that indicates how the text should be split.
#' @inheritParams get_causal_next_tokens_tbl
#'
#' @ignore_regex Can ignore certain characters when calculates the log probabilities. For example `^[[:punct:]]$` will ignore all punctuation  that stands alone in a token.
#'
#' @return a vector of log probabilities.
#'
#' @export
get_causal_log_prob <- function(x, by = rep(1, length(x)), ignore_regex = "", model = "gpt2", add_bos_token = NULL, stride = 1, config_model = NULL, config_tokenizer = NULL) {
  if (length(x) != length(by)) stop2("The argument `by` has an incorrect length.")
  if (length(x) <= 1) stop2("The argument `x` needs at least two elements.")
  x <- trimws(x, whitespace = "[ \t]")
  texts <- split(x, by)
  N <- length(texts)
  out <- tidytable::map2.(texts, names(texts), function(words, item) {
    # words <- texts[[1]]
    # item <- names(texts[1])

    ls_mat <- causal_log_prob_mat(words, model = model, add_bos_token = add_bos_token, stride = stride,  config_model = config_model, config_tokenizer = config_tokenizer)

    if(length(words) >1){
      words_lm <- c(words[1], paste0(" ", words[-1]))
    } else {
      words_lm <- words
    }
    tokens <- lapply(get_id(words_lm, model, add_bos_token, config = config_tokenizer),
                     get_tokens.numeric, model = model, add_bos_token = add_bos_token, config = config_tokenizer)
    token_n <- tidytable::map_dbl.(tokens, length)
    index_vocab <- data.table::chmatch(unlist(tokens), rownames(ls_mat[[1]]))
    message_verbose("Text id: ", item, "\n`", paste(words, collapse = " "), "`")

    token_lp <- tidytable::map2_dbl.(index_vocab, 1:ncol(ls_mat[[1]]), ~ ls_mat[[1]][.x, .y])

    if (options()$pangoling.debug) {
      print("******")
      sent <- tidytable::map_chr.(tokens, function(x) paste0(x, collapse = "|"))
      print(paste0("[", sent, "]", collapse = " "))
      print(token_lp)
    }
    if(length(ignore_regex) >0 && ignore_regex != "") {
      pos <- which(grepl(pattern = ignore_regex, x = unlist(tokens)))
      token_lp[pos] <- 0
    }
    # ignores the NA in the first column if it starts with a special character
    if(unlist(tokens)[1] %in% tokenizer(model)$all_special_tokens) token_lp[1] <- 0

    word_lp <- vector(mode="numeric", length(words))
    n <- 1
    for(i in  seq_along(token_n)){
      # i <- 1
      t <- token_n[i]
      if(n <1 || !n %in% c(cumsum(c(0,token_n))+1)) {
        word_lp[i] <- NA
      } else {
        word_lp[i] <- sum(token_lp[n:(n+(t-1))])
      }
      n <- n + t
      # i <- i + 1
    }
    word_lp
  })
  unlist(out, recursive = FALSE)
}





causal_log_prob_mat <- function(words, model = "gpt2", add_bos_token = NULL, stride = 1,config_model = NULL, config_tokenizer= NULL) {
  tkzr <- tokenizer(model, add_bos_token = add_bos_token, config_tokenizer)
  tkzr$pad_token <- tkzr$eos_token
  max_length <- tkzr$max_model_input_sizes[model]

  text <- paste0(words, collapse = " ")
  tensor <- tkzr$encode(text, return_tensors = "pt", stride = as.integer(stride), truncation = TRUE, return_overflowing_tokens = TRUE, padding = TRUE)
  # for test
  # tensor <- tkzr$encode(text, return_tensors = "pt", stride = 2L, truncation =TRUE, return_overflowing_tokens=TRUE, padding = TRUE, max_length = 3L)

  ids <- unlist(tensor$tolist())
  tensor_size <- length(ids)


  message_verbose("Processing ", tensor$shape[0], " batch(es) of ", tensor$shape[1], " tokens.")
  message_verbose("Processing using causal model '", model, "'...")

  out_lm <- lang_model(model, task = "causal", config = config_model)(tensor)

  logits_b <- out_lm$logits

  if (logits_b$shape[0] > 1) {
    # if there is a sliding window, because
    # max_tokens was exceeded:
    final_words <- lapply(1:(logits_b$shape[0] - 1), function(x) logits_b[x][seq(stride, max_length - 1)])
    logits <- torch$row_stack(c(logits_b[0], final_words))

    first_tokens <- reticulate::py_to_r(tkzr$convert_ids_to_tokens(tensor[0]))
    final_tokens <- tidytable::map_chr(1:(logits_b$shape[0] - 1), function(n) {
      t <- tensor[n][seq(stride, max_length - 1)]
      # in case the tensor is of size 1 and lost a dimension:
      if (t$shape$numel() == 1L) t <- t$reshape(1L)
      reticulate::py_to_r(tkzr$convert_ids_to_tokens(t))
    })

    tokens <- c(first_tokens, final_tokens)
  } else {
    logits <- logits_b[0]
    tokens <- tkzr$convert_ids_to_tokens(tensor[0])
  }

  lp <- reticulate::py_to_r(torch$log_softmax(logits, dim = -1L))$tolist()
  rm(logits)
  rm(logits_b)
  gc(full = TRUE)
  mat <- do.call("cbind", lp)
  # remove the last prediction, and the first is NA
  mat <- cbind(rep(NA, nrow(mat)), mat[, -ncol(mat)])
  rownames(mat) <- get_tr_vocab(model)
  colnames(mat) <- unlist(tokens)
  list(mat)
}



## #'
## #' Get a matrix with log probability of each word phrase of a vector given its previous context using a transformer model from huggingface.com
## #'
## #' @inheritParams get_causal_log_prob
## #'
## #' @return matrix
## #' @export
## #'
## #' @examples
## get_causal_log_prob_mat <- function(x, by = rep(1, length(x)), model = "gpt2", add_bos_token = NULL, stride = 1, ...) {
##   x <- trimws(x, whitespace = "[ \t]")
##   texts <- split(x, by)
##   N <- length(texts)
##   tidytable::map2.(texts, names(texts), function(words, item) {
##     causal_log_prob_mat(words, model = model, add_bos_token = add_bos_token, stride = stride, ...)
##   })
## }

