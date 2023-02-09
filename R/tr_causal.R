#' Preloads a causal language model
#'
#' Preloads a causal language model to speed up next runs.
#'
#' For more about causal models, see [chapter 7 of hugging face documentation](https://huggingface.co/course/chapter7/6).
#'
#' If not specified, the causal model that will be used is the one set in specified in the global option `pangoling.causal.default`, this can be accessed via `getOption("pangoling.causal.default")` (by default "`r getOption("pangoling.causal.default")`"). To change the default option use `options(pangoling.causal.default = "newcausalmodel")`.
#'
#' A list of possible causal models can be found in [hugging face website](https://huggingface.co/).
#'
#' Using the  `config_model` and `config_tokenizer` arguments, it's possible to control how the model and tokenizer from hugging face is accessed, see the python method [`from_pretrained`](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained) for details. In case of errors check the status of [https://status.huggingface.co/](https://status.huggingface.co/)
#'
#' @param model Name of a pretrained model stored locally on the (huggingface.co).
#' @param add_special_tokens Whether to include beginning of text special tokens. By default  acts as the [AutoTokenizer](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoTokenizer).
#' @param config_model List with other arguments that control how the model from hugging face is accessed.
#' @param config_tokenizer List with other arguments that control how the tokenizer from hugging face is accessed.
#'
#' @return Nothing.
#'
#' @examplesIf interactive()
#' causal_preload(model = "gpt2")
#'
#' @family causal model functions
#' @export
#'
causal_preload <- function(model = getOption("pangoling.causal.default"),
                           add_special_tokens = NULL,
                           config_model = NULL, config_tokenizer = NULL) {
  lang_model(model, task = "causal", config_model)
  tokenizer(model, add_special_tokens = add_special_tokens, config_tokenizer)
  invisible()
}

#' Returns the configuration of a causal model
#'
#' @inheritParams causal_preload
#' @inherit  causal_preload details
#' @return A list with the configuration of the model.
#' @examplesIf interactive()
#' causal_config(model = "gpt2")
#'
#' @family causal model functions
#' @export
causal_config <- function(model = getOption("pangoling.causal.default"), config_model = NULL) {
  lang_model(model = model,
             task = "causal",
             config_model = config_model)$config$to_dict()
}

#' Get the possible next tokens and their log probabilities its previous context using a causal transformer
#'
#' Get the possible next tokens and their log probabilities based on its previous context using a causal transformer model from [Hugging Face](https://huggingface.co).
#'
#' @section More examples:
#' See the  [online article](https://bruno.nicenboim.me/pangoling/articles/intro-gpt2.html) in pangoling website for more examples.
#'
#' @param context The context.
#' @inheritParams causal_preload
#' @inherit  causal_preload details
#' @return A table with possible next tokens and their log-probabilities.
#' @examplesIf interactive()
#' causal_next_tokens_tbl(
#'   context = "The apple doesn't fall far from the",
#'   model = "gpt2"
#' )
#'
#' @family causal model functions
#' @export
causal_next_tokens_tbl <- function(context, model = getOption("pangoling.causal.default"),
                                   add_special_tokens = NULL,
                                   config_model = NULL,
                                   config_tokenizer = NULL) {
  message_verbose("Processing using causal model '", model, "'...")

  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer)
  context_tensor <- encode(context,
                           tkzr,
                           add_special_tokens = add_special_tokens)
  generated_outputs <- lang_model(model, "causal", config_model)(context_tensor)
  n_tokens <- length(context_tensor$tolist()[0])
  logits_next_word <- generated_outputs$logits[0][n_tokens - 1]
  l_softmax <- torch$log_softmax(logits_next_word, dim = -1L)$tolist()
  lp <- reticulate::py_to_r(l_softmax) |>
    unlist()
  vocab <- get_vocab(tkzr)
  tidytable::tidytable(token = vocab, lp = lp) |>
    tidytable::arrange.(-lp)
}


#' Get the log probability of each element of a vector of words (or phrases) using a causal transformer
#'
#' Get the log probability of each element of a vector of words (or phrases) using a causal transformer model. See \code{vignette("transformer-gpt2", package = "pangoling")} for examples.
#'
#'
#' @param x Vector of words, phrases or texts.
#' @param .by Vector that indicates how the text should be split.
#' @inheritParams causal_preload
#' @param ignore_regex Can ignore certain characters when calculates the log probabilities. For example `^[[:punct:]]$` will ignore all punctuation  that stands alone in a token.
#' @inherit  causal_preload details
#' @inheritSection causal_next_tokens_tbl More examples
#' @return A named vector of log probabilities.
#'
#' @examplesIf interactive()
#' causal_lp(
#'   x = c("The", "apple", "doesn't", "fall", "far", "from", "the", "tree."),
#'   model = "gpt2"
#' )
#'
#' @family causal model functions
#' @export
causal_lp <- function(x,
                      .by = rep(1, length(x)),
                      ignore_regex = "",
                      model = getOption("pangoling.causal.default"),
                      add_special_tokens = NULL,
                      config_model = NULL,
                      config_tokenizer = NULL) {
  stride = 1 # fixed for now
  if (length(x) <= 1) stop2("The argument `x` needs at least two elements.")
  message_verbose("Processing using causal model '", model, "'...")
  word_by_word_texts <- get_word_by_word_texts(x, .by)

  pasted_texts <- lapply(word_by_word_texts,
                         function(word) paste0(word, collapse = " "))
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer)
  trf <- lang_model(model,
                    task = "causal",
                    config_model = config_model)

  tensors <- create_tensor_lst(texts = pasted_texts,
                               tkzr = tkzr,
                               add_special_tokens = add_special_tokens,
                               stride = stride)
  out <- tidytable::pmap.(list(word_by_word_texts,
                               names(word_by_word_texts), tensors),
                          function(words, item, tensor) {
    # words <- word_by_word_texts[[1]]
    # item <- names(texts[1])
    # tensor <- tensors[[1]]
    mat <- causal_mat(tensor,
                         trf,
                         tkzr,
                         add_special_tokens = add_special_tokens,
                         stride = stride)
    message_verbose("Text id: ", item, "\n`", paste(words, collapse = " "), "`")
    word_lp(words,
            mat = mat,
            ignore_regex = ignore_regex,
            model = model,
            add_special_tokens = add_special_tokens,
            config_tokenizer = config_tokenizer)
  })
  unlist(out, recursive = FALSE)
}


#' Get the log probability of each token in a sentence (or group of sentences) using a causal transformer
#'
#' Get the log probability of each token in a sentence (or group of sentences) using a causal transformer model.
#'
#'
#' @param texts Vector or list of texts.
#' @param .id Name of the column with the sentence id.
#' @inheritParams causal_preload
#' @param ignore_regex Can ignore certain characters when calculates the log probabilities. For example `^[[:punct:]]$` will ignore all punctuation  that stands alone in a token.
#' @inherit  causal_preload details
#' @inheritSection causal_next_tokens_tbl More examples
#' @return A table with token names (`token`), log-probability (`lp`) and optionally sentence id.
#'
#' @examplesIf interactive()
#' causal_tokens_lp_tbl(
#'   texts = c("The apple doesn't fall far from the tree."),
#'   model = "gpt2"
#' )
#'
#' @family causal model functions
#' @export
causal_tokens_lp_tbl <- function(texts,
                                 model = getOption("pangoling.causal.default"),
                                 add_special_tokens = NULL,
                                 config_model = NULL,
                                 config_tokenizer = NULL,
                                 .id = NULL) {
  stride = 1
  message_verbose("Processing using causal model '", model, "'...")
  ltexts <- as.list(unlist(texts, recursive = TRUE))
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer)
  trf <- lang_model(model,
                    task = "causal",
                    config_model = config_model)
  tensors <- create_tensor_lst(ltexts,
                               tkzr,
                              add_special_tokens = add_special_tokens,
                              stride = stride )

  ls_mat <- tidytable::map.(tensors, function(tensor) {
    causal_mat(tensor,
               trf,
               tkzr,
               add_special_tokens = add_special_tokens,
               stride = stride)
  })
  lindex_vocab <- tokenize(unlist(texts, recursive = TRUE),
    model = model,
    add_special_tokens = add_special_tokens,
    config = config_tokenizer
  )

  tidytable::map2_dfr.(lindex_vocab, ls_mat, function(vocab, mat) {
    tidytable::tidytable(token = vocab,
                         lp = tidytable::map2_dbl.(vocab, 1:ncol(mat), ~ mat[.x, .y]))
  }, .id = .id)
}


#' @noRd
causal_mat <- function(tensor,
                       trf,
                       tkzr,
                       add_special_tokens = NULL,
                       stride = 1) {
  message_verbose("Processing ",
                  tensor$shape[0],
                  " batch(es) of ",
                  tensor$shape[1], " tokens.")
  logits_b <- trf(tensor)$logits

  if (logits_b$shape[0] > 1) {
    stop2("Input is too long, exceeding")
    # # if there is a sliding window, because
    # # max_tokens was exceeded:
    # final_words <- lapply(1:(logits_b$shape[0] - 1), function(x) logits_b[x][seq(stride, max_length - 1)])
    # logits <- torch$row_stack(c(logits_b[0], final_words))
    #
    # first_tokens <- tkzr$convert_ids_to_tokens(tensor[0])
    # final_tokens <- tidytable::map(0:(logits_b$shape[0] - 1), function(n) {
    #   t <- tensor[n][seq(stride, max_length - 1)]
    #   # in case the tensor is of size 1 and lost a dimension:
    #   if (t$shape$numel() == 1L) t <- t$reshape(1L)
    #   tkzr$convert_ids_to_tokens(t)
    # }) |>
    #   unlist()
    #
    # tokens <- c(first_tokens, final_tokens)
  }
  logits <- logits_b[0]
  tokens <- tkzr$convert_ids_to_tokens(tensor[0])

  lp <- reticulate::py_to_r(torch$log_softmax(logits, dim = -1L))$tolist()
  rm(logits)
  rm(logits_b)
  gc(full = TRUE)
  mat <- do.call("cbind", lp)
  # remove the last prediction, and the first is NA
  mat <- cbind(rep(NA, nrow(mat)), mat[, -ncol(mat)])
  rownames(mat) <- get_vocab(tkzr)
  colnames(mat) <- unlist(tokens)
  mat
}


#' Get a list of matrices with the log probabilities of possible word given its previous context using a causal transformer
#'
#' Get a list of matrices with the log probabilities of possible word given its previous context using a causal transformer model.
#'
#' @inheritParams causal_lp
#' @param ignore_regex Can ignore certain characters when calculates the log probabilities. For example `^[[:punct:]]$` will ignore all punctuation  that stands alone in a token.
#' @inherit  causal_preload details
#' @inheritSection causal_next_tokens_tbl More examples
#' @return A list of matrices with tokens in their columns and the vocabulary of the model in their rows
#'
#' @examplesIf interactive()
#' causal_lp_mats(
#'   x = c("The", "apple", "doesn't", "fall", "far", "from", "the", "tree."),
#'   model = "gpt2"
#' )
#'
#' @family causal model functions
#' @export
#'
causal_lp_mats <- function(x,
                          .by = rep(1, length(x)),
                          model = getOption("pangoling.causal.default"),
                          add_special_tokens = NULL,
                          config_model = NULL,
                          config_tokenizer = NULL) {
  stride = 1
  message_verbose("Processing using causal model '", model, "'...")
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer)
  trf <- lang_model(model,
                    task = "causal",
                    config_model = config_model)
  x <- trimws(x, whitespace = "[ \t]")
  word_by_word_texts <- split(x, .by)
  pasted_texts <- lapply(word_by_word_texts,
                         function(word) paste0(word, collapse = " "))
  tensors <- create_tensor_lst(pasted_texts,
                               tkzr,
                               add_special_tokens = add_special_tokens,
                               stride = stride)
  lmat <- tidytable::pmap.(list(word_by_word_texts, names(word_by_word_texts), tensors),
                   function(words, item, tensor) {
    causal_mat(tensor,
               trf,
               tkzr,
               add_special_tokens = add_special_tokens,
               stride = stride)
  })
}
