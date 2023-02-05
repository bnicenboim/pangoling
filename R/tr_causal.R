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

  tidytable::tidytable(token = vocab, log_prob = lp) |>
    tidytable::arrange.(-log_prob)
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
get_causal_log_prob <- function(x, .by = rep(1, length(x)), ignore_regex = "", model = "gpt2", add_special_tokens = NULL, stride = 1, config_model = NULL, config_tokenizer = NULL) {
  if (length(x) != length(.by)) stop2("The argument `.by` has an incorrect length.")
  if (length(x) <= 1) stop2("The argument `x` needs at least two elements.")
  message_verbose("Processing using causal model '", model, "'...")
  x <- trimws(x, whitespace = "[ \t]")
  word_by_word_texts <- split(x, .by)
  N <- length(word_by_word_texts)
  pasted_texts <- lapply(word_by_word_texts, function(word) paste0(word, collapse = " "))
  tensors <- create_causal_tensor_lst(pasted_texts, model = model, add_special_tokens = add_special_tokens, stride = stride, config = config_tokenizer)
  out <- tidytable::pmap.(list(word_by_word_texts, names(word_by_word_texts), tensors), function(words, item, tensor) {
    # words <- word_by_word_texts[[1]]
    # item <- names(texts[1])
    # tensor <- tensors[[1]]
    ls_mat <- causal_log_prob_mat(tensor, model = model, add_special_tokens = add_special_tokens, stride = stride, config_model = config_model, config_tokenizer = config_tokenizer)

    if (length(words) > 1) {
      words_lm <- c(words[1], paste0(" ", words[-1]))
    } else {
      words_lm <- words
    }
    tokens <- lapply(get_id(words_lm, model, add_special_tokens = add_special_tokens, config = config_tokenizer),
      get_tokens.numeric,
      model = model, add_special_tokens = add_special_tokens, config = config_tokenizer
    )
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
    if (length(ignore_regex) > 0 && ignore_regex != "") {
      pos <- which(grepl(pattern = ignore_regex, x = unlist(tokens)))
      token_lp[pos] <- 0
    }
    # ignores the NA in the first column if it starts with a special character
    if (unlist(tokens)[1] %in% tokenizer(model)$all_special_tokens) token_lp[1] <- 0

    word_lp <- vector(mode = "numeric", length(words))
    n <- 1
    for (i in seq_along(token_n)) {
      # i <- 1
      t <- token_n[i]
      if (n < 1 || !n %in% c(cumsum(c(0, token_n)) + 1)) {
        word_lp[i] <- NA
      } else {
        word_lp[i] <- sum(token_lp[n:(n + (t - 1))])
      }
      n <- n + t
      # i <- i + 1
    }
    names(word_lp) <- words
    word_lp
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
get_causal_tokens_log_prob_tbl <- function(texts, model = "gpt2", add_special_tokens = NULL, stride = 1, config_model = NULL, config_tokenizer = NULL, .id = NULL) {
  message_verbose("Processing using causal model '", model, "'...")
  ltexts <- as.list(unlist(texts, recursive = TRUE))

  tensors <- create_causal_tensor_lst(ltexts,
    model = model,
    add_special_tokens = add_special_tokens,
    stride = stride, config = config_tokenizer
  )
  lls_mat <- tidytable::map.(tensors, function(tensor) {
    causal_log_prob_mat(tensor, model = model, add_special_tokens = add_special_tokens, stride = stride, config_model = config_model, config_tokenizer = config_tokenizer)
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
create_causal_tensor_lst <- function(texts, model = "gpt2", add_special_tokens = NULL, stride = 1, config = NULL) {
  tkzr <- tokenizer(model, add_special_tokens = add_special_tokens, config)
  tkzr$pad_token <- tkzr$eos_token
  max_length <- tkzr$max_len_single_sentence
  if (is.null(max_length) || is.na(max_length) || max_length < 1) {
    warning("Unknown maximum length of input. This might cause a problem for long inputs exceeding the maximum length.")
    max_length <- Inf
  }
  # text <- paste0(words, collapse = " ")
  if (is.null(add_special_tokens)) {
    lapply(texts, function(text) {
      tensor <- tkzr$encode(text, return_tensors = "pt", stride = as.integer(stride), truncation = is.finite(max_length), return_overflowing_tokens = is.finite(max_length), padding = is.finite(max_length))
      tensor
    })
  } else {
    lapply(texts, function(text) {
      tensor <- tkzr$encode(text, return_tensors = "pt", stride = as.integer(stride), truncation = is.finite(max_length), return_overflowing_tokens = is.finite(max_length), padding = is.finite(max_length), add_special_tokens = add_special_tokens)
      tensor
    })
  }
}

#' @noRd
causal_log_prob_mat <- function(tensor, model = "gpt2", add_special_tokens = NULL, stride = 1, config_model = NULL, config_tokenizer = NULL) {
  tkzr <- tokenizer(model, add_special_tokens = add_special_tokens, config = config_tokenizer)

  # for test
  # tensor <- tkzr$encode(text, return_tensors = "pt", stride = 2L, truncation =TRUE, return_overflowing_tokens=TRUE, padding = TRUE, max_length = 3L)

  ids <- unlist(tensor$tolist())
  tensor_size <- length(ids)

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
#' @inheritParams get_causal_log_prob
#'
#' @return A matrix.
#' @export
#'
get_causal_log_prob_mat <- function(x, .by = rep(1, length(x)), model = "gpt2", add_special_tokens = NULL, stride = 1, config_model = NULL, config_tokenizer = NULL) {
  message_verbose("Processing using causal model '", model, "'...")

  x <- trimws(x, whitespace = "[ \t]")
  word_by_word_texts <- split(x, .by)
  N <- length(word_by_word_texts)
  pasted_texts <- lapply(word_by_word_texts, function(word) paste0(word, collapse = " "))
  tensors <- create_causal_tensor_lst(pasted_texts, model = model, add_special_tokens = add_special_tokens, stride = stride, config = config_tokenizer)
  tidytable::pmap.(list(word_by_word_texts, names(word_by_word_texts), tensors), function(words, item, tensor) {
    causal_log_prob_mat(tensor, model = model, add_special_tokens = add_special_tokens, stride = stride, config_model = config_model, config_tokenizer = config_tokenizer)
  })
}

#' Calculates perplexity
#'
#' Calculates perplexity of a vector of (log-)probabilities.
#'
#' If x are raw probabilities (NOT the default), then perplexity is calculated as follows:
#'
#' \deqn{\left(\prod_n x_n \right)^\frac{1}{N}
#'
#' @param x	A vector of log-probabilities.
#' @param na.rm	Should missing values (including NaN) be removed?
#' @param log.p If TRUE (default),  x are assumed to be log-transformed probabilities with base e, if FALSE x are assumed to be raw probabilities, alternatively log.p can be the base of other logarithmic transformations.
#' @return The perplexity
#'
#' @examples
#' probs <- c(.3, .5, .6)
#' perplexity(probs, log.p = FALSE)
#' lprobs <- log(probs)
#' perplexity(lprobs, log.p = TRUE)
#' @export
#'
perplexity <- function(x, na.rm = FALSE, log.p
                       = TRUE) {
  if (log.p == FALSE) {
    prod(x, na.rm = na.rm)^(-1 / length(x))
  } else if (log.p || all.equal(log.p, exp(1))) {
    exp(-sum(x, na.rm = na.rm) / length(x))
  } else {
    log.p^(-sum(x, na.rm = na.rm) / length(x))
  }
}