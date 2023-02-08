#' Preloads a causal language model
#'
#' Preloads a causal language model to speed up next runs.
#'
#' @inheritParams get_causal_next_tokens_tbl
#'
#' @return Nothing.
#' @export
#'
preload_causal <- function(model = "gpt2", add_special_tokens = NULL, config_model = NULL, config_tokenizer = NULL) {
  lang_model(model, task = "causal", config_model)
  tokenizer(model, add_special_tokens = add_special_tokens, config_tokenizer)
  invisible()
}

#' Returns the configuration of a causal model
#'
#' @inheritParams get_causal_next_tokens_tbl
#'
#' @return A list with the configuration of the model
#' @export
get_causal_model_config <- function(model = "gpt2", config = NULL) {
  lang_model(model = model, task = "causal", config = config)$config$to_dict()
}

#' Get the possible next tokens and their log probabilities its previous context using a causal transformer
#'
#' Get the possible next tokens and their log probabilities its previous context using a causal transformer model from [Hugging Face](https://huggingface.co).  Get the log probability of each word phrase of a vector given its previous context using a transformer model from huggingface.co/. See \code{vignette("transformer-gpt2", package = "pangoling")} for examples.
#'
#' For more about causal models, see (https://huggingface.co/course/chapter7/6). Using the  `config_model` and `config_tokenizer` arguments, it's possible to control how the model and tokenizer from hugging face is accessed, see [from_pretrained](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained) for details. In case of errors check the status of https://status.huggingface.co/
#'
#'
#' @param context Context.
#' @param model Name of a pretrained model stored on the huggingface.co. (Maybe a path to a  model (.pt or .bin file) stored locally will work.)
#' @param add_special_tokens Whether to include beginning of text special tokens. By default  acts as the [AutoTokenizer](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoTokenizer).
#' @param config_model List with other arguments that control how the model from hugging face is accessed.
#' @param config_tokenizer List with arguments that control how the tokenizer from hugging face is accessed.
#'
#' @return A table with possible next tokens and their log-probabilities.
#' @export
get_causal_next_tokens_tbl <- function(context, model = "gpt2", add_special_tokens = NULL, config_model = NULL, config_tokenizer = NULL) {
  message_verbose("Processing using causal model '", model, "'...")

  tkzr <- tokenizer(model, add_special_tokens = add_special_tokens, config_tokenizer)
  context_tensor <- tkzr$encode(context, return_tensors = "pt")
  generated_outputs <- lang_model(model, "causal", config_model)(context_tensor)
  n_tokens <- length(context_tensor$tolist()[0])
  logits_next_word <- generated_outputs$logits[0][n_tokens - 1]
  lp <- reticulate::py_to_r(torch$log_softmax(logits_next_word, dim = -1L)$tolist()) |>
    unlist()

  vocab <- get_tr_vocab(model, add_special_tokens = add_special_tokens, config_tokenizer)

  tidytable::tidytable(token = vocab, lp = lp) |>
    tidytable::arrange.(-lp)
}


#' Get the log probability of each element of a vector of words (or phrases) using a causal transformer
#'
#' Get the log probability of each element of a vector of words (or phrases) using a causal transformer model. See \code{vignette("transformer-gpt2", package = "pangoling")} for examples.
#'
#' For more about causal models, see (https://huggingface.co/course/chapter7/6).  Using the  `config_model` and `config_tokenizer` arguments, it's possible to control how the model and tokenizer from hugging face is accessed, see [from_pretrained](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained) for details. In case of errors check the status of https://status.huggingface.co/
#'
#' @param x Vector of words, phrases or texts.
#' @param .by Vector that indicates how the text should be split.
#' @inheritParams get_causal_next_tokens_tbl
#' @param ignore_regex Can ignore certain characters when calculates the log probabilities. For example `^[[:punct:]]$` will ignore all punctuation  that stands alone in a token.
#'
#' @return A named vector of log probabilities.
#'
#' @export
get_causal_lp <- function(x, .by = rep(1, length(x)), ignore_regex = "", model = "gpt2", add_special_tokens = NULL, stride = 1, config_model = NULL, config_tokenizer = NULL) {

  if (length(x) <= 1) stop2("The argument `x` needs at least two elements.")

  message_verbose("Processing using causal model '", model, "'...")

  word_by_word_texts <- get_word_by_word_texts(x, .by)

  # N <- length(word_by_word_texts)
  pasted_texts <- lapply(word_by_word_texts, function(word) paste0(word, collapse = " "))
  tensors <- create_tensor_lst(pasted_texts, model = model, add_special_tokens = add_special_tokens, stride = stride, config = config_tokenizer)
  out <- tidytable::pmap.(list(word_by_word_texts, names(word_by_word_texts), tensors), function(words, item, tensor) {
    # words <- word_by_word_texts[[1]]
    # item <- names(texts[1])
    # tensor <- tensors[[1]]
    ls_mat <- causal_lp_mat(tensor, model = model, add_special_tokens = add_special_tokens, stride = stride, config_model = config_model, config_tokenizer = config_tokenizer)
    message_verbose("Text id: ", item, "\n`", paste(words, collapse = " "), "`")
 word_lp(words, mat = ls_mat[[1]],ignore_regex = ignore_regex,model = model, add_special_tokens = add_special_tokens, config_tokenizer = config_tokenizer )
  })
  unlist(out, recursive = FALSE)
}


#' Get the log probability of each token in a sentence (or group of sentences) using a causal transformer
#'
#' Get the log probability of each token in a sentence (or group of sentences) using a causal transformer model. See \code{vignette("transformer-gpt2", package = "pangoling")} for examples.
#'
#' For more about causal models, see (https://huggingface.co/course/chapter7/6).  Using the  `config_model` and `config_tokenizer` arguments, it's possible to control how the model and tokenizer from hugging face is accessed, see [from_pretrained](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained) for details. In case of errors check the status of https://status.huggingface.co/
#'
#' @param texts Vector or list of texts.
#' @inheritParams get_causal_next_tokens_tbl
#' @param .id Column with sentence id.
#'
#' @return A table with token names, log-probability and optionally sentence id.
#'
#' @export
get_causal_tokens_lp_tbl <- function(texts, model = "gpt2", add_special_tokens = NULL, stride = 1, config_model = NULL, config_tokenizer = NULL, .id = NULL) {
  message_verbose("Processing using causal model '", model, "'...")
  ltexts <- as.list(unlist(texts, recursive = TRUE))

  tensors <- create_tensor_lst(ltexts,
    model = model,
    add_special_tokens = add_special_tokens,
    stride = stride, config = config_tokenizer
  )
  lls_mat <- tidytable::map.(tensors, function(tensor) {
    causal_lp_mat(tensor, model = model, add_special_tokens = add_special_tokens, stride = stride, config_model = config_model, config_tokenizer = config_tokenizer)
  })
  lindex_vocab <- get_tokens(unlist(texts, recursive = TRUE),
    model = model,
    add_special_tokens = add_special_tokens, config = config_tokenizer
  )

  tidytable::map2_dfr.(lindex_vocab, lls_mat, function(vocab, ls_mat) {
    tidytable::tidytable(token = vocab, lprob = tidytable::map2_dbl.(vocab, 1:ncol(ls_mat[[1]]), ~ ls_mat[[1]][.x, .y]))
  }, .id = .id)
}


#' @noRd
causal_lp_mat <- function(tensor, model = "gpt2", add_special_tokens = NULL, stride = 1, config_model = NULL, config_tokenizer = NULL) {
  tkzr <- tokenizer(model, add_special_tokens = add_special_tokens, config = config_tokenizer)

  message_verbose("Processing ", tensor$shape[0], " batch(es) of ", tensor$shape[1], " tokens.")

  out_lm <- lang_model(model, task = "causal", config = config_model)(tensor)

  logits_b <- out_lm$logits

  if (logits_b$shape[0] > 1) {
    # if there is a sliding window, because
    # max_tokens was exceeded:
    final_words <- lapply(1:(logits_b$shape[0] - 1), function(x) logits_b[x][seq(stride, max_length - 1)])
    logits <- torch$row_stack(c(logits_b[0], final_words))

    first_tokens <- tkzr$convert_ids_to_tokens(tensor[0])
    final_tokens <- tidytable::map(0:(logits_b$shape[0] - 1), function(n) {
      t <- tensor[n][seq(stride, max_length - 1)]
      # in case the tensor is of size 1 and lost a dimension:
      if (t$shape$numel() == 1L) t <- t$reshape(1L)
      tkzr$convert_ids_to_tokens(t)
    }) |>
      unlist()

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
  rownames(mat) <- get_tr_vocab(model, add_special_tokens = add_special_tokens, config = config_tokenizer)
  colnames(mat) <- unlist(tokens)
  list(mat)
}


#' Get a matrix with the log probabilities of possible word given its previous context using a causal transformer
#'
#' Get a matrix with the log probabilities of possible word given its previous context using a causal transformer model from [Hugging Face](https://huggingface.co).
#'
#' For more about causal models, see (https://huggingface.co/course/chapter7/6). Using the  `config_model` and `config_tokenizer` arguments, it's possible to control how the model and tokenizer from hugging face is accessed, see [from_pretrained](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained) for details. In case of errors check the status of https://status.huggingface.co/
#'
#'
#'
#' @inheritParams get_causal_lp
#'
#' @return A matrix.
#' @export
#'
get_causal_lp_mat <- function(x, .by = rep(1, length(x)), model = "gpt2", add_special_tokens = NULL, stride = 1, config_model = NULL, config_tokenizer = NULL) {
  message_verbose("Processing using causal model '", model, "'...")

  x <- trimws(x, whitespace = "[ \t]")
  word_by_word_texts <- split(x, .by)
  # N <- length(word_by_word_texts)
  pasted_texts <- lapply(word_by_word_texts, function(word) paste0(word, collapse = " "))
  tensors <- create_tensor_lst(pasted_texts, model = model, add_special_tokens = add_special_tokens, stride = stride, config = config_tokenizer)
  tidytable::pmap.(list(word_by_word_texts, names(word_by_word_texts), tensors), function(words, item, tensor) {
    causal_lp_mat(tensor, model = model, add_special_tokens = add_special_tokens, stride = stride, config_model = config_model, config_tokenizer = config_tokenizer)
  })
}

