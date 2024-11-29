#' Preloads a causal language model
#'
#' Preloads a causal language model to speed up next runs.
#'
#' A causal language model (also called GPT-like, auto-regressive, or decoder
#' model) is a type of large language model usually used for text-generation
#' that can predict the next word (or more accurately in fact token) based
#' on a preceding context.
#'
#' If not specified, the causal model that will be used is the one set in
#' specified in the global option `pangoling.causal.default`, this can be
#' accessed via `getOption("pangoling.causal.default")` (by default
#' "`r getOption("pangoling.causal.default")`"). To change the default option
#' use `options(pangoling.causal.default = "newcausalmodel")`.
#'
#' A list of possible causal models can be found in
#' [Hugging Face website](https://huggingface.co/models?pipeline_tag=text-generation).
#'
#' Using the  `config_model` and `config_tokenizer` arguments, it's possible to
#'  control how the model and tokenizer from Hugging Face is accessed, see the
#'  Python method
#'  [`from_pretrained`](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained)
#'  for details.
#'
#'  In case of errors when a new model is run, check the status of
#'  [https://status.huggingface.co/](https://status.huggingface.co/)
#'
#' @param model Name of a pre-trained model or folder.
#' @param checkpoint Folder of a checkpoint.
#' @param add_special_tokens Whether to include special tokens. It has the
#'                           same default as the
#'                           [AutoTokenizer](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoTokenizer) method in Python.
#' @param config_model List with other arguments that control how the
#'                      model from Hugging Face is accessed.
#' @param config_tokenizer List with other arguments that control how the tokenizer from Hugging Face is accessed.
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
                           checkpoint = NULL,
                           add_special_tokens = NULL,
                           config_model = NULL, config_tokenizer = NULL) {
  message_verbose("Preloading causal model ", model, "...")
  lang_model(model, checkpoint = checkpoint, task = "causal", config_model = config_model)
  tokenizer(model, add_special_tokens = add_special_tokens, config_tokenizer = config_tokenizer)
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
causal_config <- function(model = getOption("pangoling.causal.default"),
                          checkpoint = NULL, config_model = NULL) {
  lang_model(
    model = model,
    checkpoint = checkpoint,
    task = "causal",
    config_model = config_model
  )$config$to_dict()
}

#' Get the possible next tokens and their log probabilities its previous context using a causal transformer
#'
#' Get the possible next tokens and their log probabilities based on its
#' previous context using a causal transformer model from [Hugging Face](https://huggingface.co).
#'
#' @section More examples:
#' See the
#' [online article](https://bruno.nicenboim.me/pangoling/articles/intro-gpt2.html)
#' in pangoling website for more examples.
#'
#' @param l_context The left context.
#' @inheritParams causal_preload
#' @inherit  causal_preload details
#' @return A table with possible next tokens and their log-probabilities.
#' @examplesIf interactive()
#' causal_next_tokens_pred_tbl(
#'   context = "The apple doesn't fall far from the",
#'   model = "gpt2"
#' )
#'
#' @family causal model functions
#' @export
causal_next_tokens_pred_tbl <- function(l_context,
                                        log.p = getOption("pangoling.log.p"),
                                        model = getOption("pangoling.causal.default"),
                                        checkpoint = NULL,
                                        add_special_tokens = NULL,
                                        config_model = NULL,
                                        config_tokenizer = NULL) {
  if (length(unlist(l_context)) > 1) stop2("Only one context is allowed in this function.")
  message_verbose_model(model, checkpoint)
  trf <- lang_model(model,
                    checkpoint = checkpoint,
                    task = "causal",
                    config_model = config_model
                    )
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer
                    )

  # no batches allowed
  context_tensor <- encode(list(unlist(l_context)),
                           tkzr,
                           add_special_tokens = add_special_tokens
                           )$input_ids
  generated_outputs <- trf(context_tensor)
  n_tokens <- length(context_tensor$tolist()[0])
  logits_next_word <- generated_outputs$logits[0][n_tokens - 1]
  l_softmax <- torch$log_softmax(logits_next_word, dim = -1L)$tolist()
  lp <- reticulate::py_to_r(l_softmax) |>
    unlist()
  vocab <- get_vocab(tkzr)
  tidytable::tidytable(token = vocab,
                       pred = lp |> ln_p_change(log.p = log.p)) |>
    tidytable::arrange(-pred)
}


#' Get the predictability of each element of a vector of words (or phrases) in a series of texts using a causal transformer
#'
#' Get the predictability (by default the natural logarithm of the word probability) of each element of a vector of words (or phrases) in a series of texts using a causal transformer model. See the
#' [online article](https://bruno.nicenboim.me/pangoling/articles/intro-gpt2.html)
#' in pangoling website for more examples.
#'
#'
#' @param x Vector of words, phrases or texts.
#' @param by Vector that indicates how the text should be split.
#' @param sep Character indicating how words are separated in a sentence.
#' @param log.p Base of the logarithm used for the output predictability values.
#'              If `TRUE` (default), the natural logarithm (base *e*) is used.
#'              If `FALSE`, the raw probabilities are returned.
#'              Alternatively, `log.p` can be set to a numeric value specifying
#'              the base of the logarithm (e.g., `2` for base-2 logarithms).
#' @param ... not in use.
#' @inheritParams causal_preload
#' @param ignore_regex Can ignore certain characters when calculates the log
#'                      probabilities. For example `^[[:punct:]]$` will ignore
#'                      all punctuation  that stands alone in a token.
#' @param batch_size Maximum size of the batch. Larges batches speedup
#'                   processing but take more memory.
#' @inherit  causal_preload details
#' @inheritSection causal_next_tokens_pred_tbl More examples
#' @return A named vector of log probabilities.
#'
#' @examplesIf interactive()
#' example_data <- tribble(
#'    ~sent_n, ~word,
#'        1,  "The",
#'        1,  "apple",
#'        1,  "doesn't",
#'        1,  "fall",
#'        1,  "far",
#'        1,  "from",
#'        1,  "the",
#'        1,  "tree.",
#'        2,  "Don't",
#'        2,  "judge",
#'        2,  "a",
#'        2,  "book",
#'        2,  "by",
#'        2,  "its",
#'        2,  "cover."
#' )
#' causal_words_pred(
#'   x = example_data$word,
#'   by = example_data$sent_n,
#'   model = "gpt2"
#' )
#' causal_words_pred(
#'   x = example_data$word,
#'   by = example_data$sent_n,
#'   model = "gpt2",
#'   log.p = 1/2  # surprisal values in bits (-log2(prob) = log(prob, base = 1/2))
#' )
#'
#' @family causal model functions
#' @export
causal_words_pred <- function(x,
                              by = rep(1, length(x)),
                              sep = " ",
                              log.p = getOption("pangoling.log.p"),
                              ignore_regex = "",
                              model = getOption("pangoling.causal.default"),
                              checkpoint = NULL,
                              add_special_tokens = NULL,
                              config_model = NULL,
                              config_tokenizer = NULL,
                              batch_size = 1,
                              ...) {
  dots <- list(...)
  # Check for unknown arguments
  if (length(dots) > 0) {
    unknown_args <- setdiff(names(dots), ".by")
    if (length(unknown_args) > 0) {
      stop("Unknown arguments: ", paste(unknown_args, collapse = ", "), ".")
    }
  }
  if (length(x) != length(by)) stop2("The argument `by` has an incorrect length.")

 if(any(x != trimws(x)) & sep == " ") {
   message_verbose('Notice that some words have white spaces, argument `sep` should probably set to "".')
 }
  
  stride <- 1 # fixed for now
  message_verbose_model(model, checkpoint = checkpoint)

  word_by_word_texts <- split(x, by, drop = TRUE)


  pasted_texts <- conc_words(word_by_word_texts, sep = sep)
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer)
  trf <- lang_model(model,
                    checkpoint = checkpoint,
                    task = "causal",
                    config_model = config_model)
  tensors <- create_tensor_lst(
    texts = unname(pasted_texts),
    tkzr = tkzr,
    add_special_tokens = add_special_tokens,
    stride = stride,
    batch_size = batch_size
  )

  lmats <- lapply(tensors, function(tensor) {
    causal_mat(tensor,
               trf,
               tkzr,
               add_special_tokens = add_special_tokens,
               stride = stride)
  }) |>
    unlist(recursive = FALSE)

  out <- tidytable::pmap(
                      list(
                        word_by_word_texts,
                        names(word_by_word_texts),
                        lmats
                      ),
                      function(words, item, mat) {
                        # words <- word_by_word_texts[[1]]
                        # item <- names(word_by_word_texts)[[1]]
                        # mat <- lmats[[1]]

                        message_verbose(
                          "Text id: ", item, "\n`",
                          paste(words, collapse = sep),
                          "`"
                        )
                        word_lp(words,
                                sep = sep,
                                mat = mat,
                                ignore_regex = ignore_regex,
                                model = model,
                                add_special_tokens = add_special_tokens,
                                config_tokenizer = config_tokenizer)
                      }
                    )

  lps <- out |> unsplit(by, drop = TRUE)
  names(lps) <- out |> lapply(function(x) paste0(names(x),"")) |>
    unsplit(by, drop = TRUE)
  lps |> ln_p_change(log.p = log.p)
}



#' Get the predictability of each token in a sentence (or group of sentences) using a causal transformer
#'
#' Get the predictability (by default the natural logarithm of the word probability) of each token in a sentence (or group of sentences) using a causal transformer model.
#'
#'
#' @param texts Vector or list of texts.
#' @param .id Name of the column with the sentence id.
#' @inheritParams causal_preload
#' @inheritParams causal_words_pred
#' @inherit  causal_preload details
#' @inheritSection causal_next_tokens_pred_tbl More examples
#' @return A table with token names (`token`), predictability (`pred`) and optionally sentence id.
#'
#' @examplesIf interactive()
#' causal_tokens_pred_tbl(
#'   texts = c("The apple doesn't fall far from the tree."),
#'   model = "gpt2"
#' )
#'
#' @family causal model functions
#' @export
causal_tokens_pred_tbl <- function(texts,
                                   log.p = getOption("pangoling.log.p"),
                                   model = getOption("pangoling.causal.default"),
                                   checkpoint = NULL,
                                   add_special_tokens = NULL,
                                   config_model = NULL,
                                   config_tokenizer = NULL,
                                   batch_size = 1,
                                   .id = NULL) {
  stride <- 1
  message_verbose_model(model, checkpoint)
  ltexts <- as.list(unlist(texts, recursive = TRUE))
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer)
  trf <- lang_model(model,
                    checkpoint = checkpoint,
                    task = "causal",
                    config_model = config_model)
  tensors <- create_tensor_lst(ltexts,
                               tkzr,
                               add_special_tokens = add_special_tokens,
                               stride = stride,
                               batch_size = batch_size)

  ls_mat <- tidytable::map(tensors, function(tensor) {
    causal_mat(tensor,
               trf,
               tkzr,
               add_special_tokens = add_special_tokens,
               stride = stride)
  }) |>
    unlist(recursive = FALSE)

  tidytable::map_dfr(ls_mat, function(mat) {
    if (ncol(mat) == 1 && colnames(mat) == "") {
      tidytable::tidytable(
                   token = "",
                   pred = NA_real_
                 )
    } else {
      tidytable::tidytable(
                   token = colnames(mat),
                   pred = tidytable::map2_dbl(colnames(mat), seq_len(ncol(mat)), ~ mat[.x, .y]) |>
                     ln_p_change(log.p = log.p)
                 )
    }
  }, .id = .id)
}


#' @noRd
causal_mat <- function(tensor,
                       trf,
                       tkzr,
                       add_special_tokens = NULL,
                       stride = 1) {
  message_verbose(
    "Processing a batch of size ",
    tensor$input_ids$shape[0],
    " with ",
    tensor$input_ids$shape[1], " tokens."
  )

  if (tensor$input_ids$shape[1] == 0) {
    warning("No tokens found.", call. = FALSE)
    vocab <- get_vocab(tkzr)
    mat <- matrix(rep(NA, length(vocab)), ncol = 1)
    rownames(mat) <- vocab
    colnames(mat) <- ""
    return(list(mat))
  }

  logits_b <- trf$forward(
                    input_ids = tensor$input_ids,
                    attention_mask = tensor$attention_mask
                  )$logits
  # if (logits_b$shape[0] > 1) {
  # stop2("Input is too long")
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
  # }
  lmat <- lapply(seq_len(logits_b$shape[0]) - 1, function(i) {
    real_token_pos <- seq_len(sum(tensor$attention_mask[i]$tolist())) - 1
    logits <- logits_b[i][real_token_pos]
    # in case it's only one token, it needs to be unsqueezed
    ids <- tensor$input_ids[i]$unsqueeze(1L)
    tokens <- tkzr$convert_ids_to_tokens(ids[real_token_pos])
    lp <- reticulate::py_to_r(torch$log_softmax(logits, dim = -1L))$tolist()
    rm(logits)
    gc(full = TRUE)
    if (is.list(lp)) {
      mat <- do.call("cbind", lp)
    } else {
      # In case it's only one token, lp won't be a list
      mat <- matrix(lp, ncol = 1)
    }
    # remove the last prediction, and the first is NA
    mat <- cbind(rep(NA, nrow(mat)), mat[, -ncol(mat)])
    rownames(mat) <- get_vocab(tkzr)
    colnames(mat) <- unlist(tokens)
    mat
  })
  rm(logits_b)
  lmat
}



#' Get a list of matrices with the predictability of possible word given its previous context using a causal transformer
#'
#' Get a list of matrices with the predictability (by default the natural logarithm of the word probability) of possible word given
#' its previous context using a causal transformer model.
#'
#' @inheritParams causal_words_pred
#' @inheritParams causal_preload
#' @param sorted When default FALSE it will retain the order of groups we are splitting on. When TRUE then sorted (according to `by`) list(s) are returned. 
#' @inherit  causal_preload details
#' @inheritSection causal_next_tokens_pred_tbl More examples
#' @return A list of matrices with tokens in their columns and the vocabulary of the model in their rows
#'
#' @examplesIf interactive()
#' causal_pred_mats(
#'   x = c("The", "apple", "doesn't", "fall", "far", "from", "the", "tree."),
#'   model = "gpt2"
#' )
#'
#' @family causal model functions
#' @export
#'
causal_pred_mats <- function(x,
                             by = rep(1, length(x)),
                             sep = " ",
                             log.p = getOption("pangoling.log.p"),
                             sorted = FALSE,
                             model = getOption("pangoling.causal.default"),
                             checkpoint = NULL,
                             add_special_tokens = NULL,
                             config_model = NULL,
                             config_tokenizer = NULL,
                             batch_size = 1,
                             ...) {
  dots <- list(...)
  # Check for the deprecated .by argument
  if (!is.null(dots$.by)) {
    warning("The '.by' argument is deprecated. Please use 'by' instead.")
    by <- dots$.by # Assume that if .by is supplied, it takes precedence
  }
  # Check for unknown arguments
  if (length(dots) > 0) {
    unknown_args <- setdiff(names(dots), ".by")
    if (length(unknown_args) > 0) {
      stop("Unknown arguments: ", paste(unknown_args, collapse = ", "), ".")
    }
  }
  stride <- 1
  message_verbose_model(model, checkpoint)
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer
                    )
  trf <- lang_model(model,
                    checkpoint = checkpoint,
                    task = "causal",
                    config_model = config_model
                    )
  x <- trimws(x, whitespace = "[ \t]")
  word_by_word_texts <- split(x, by)
  pasted_texts <- conc_words(word_by_word_texts, sep = sep)
  tensors <- create_tensor_lst(unname(pasted_texts),
                               tkzr,
                               add_special_tokens = add_special_tokens,
                               stride = stride,
                               batch_size = batch_size
                               )
  lmat <- tidytable::map(
                       tensors,
                       function(tensor) {
                         causal_mat(tensor,
                                    trf,
                                    tkzr,
                                    add_special_tokens = add_special_tokens,
                                    stride = stride
                                    )
                       }
                     )
  names(lmat) <- levels(as.factor(by))
  if(!sorted) lmat <- lmat[unique(as.factor(by))]
  lmat |>
    unlist(recursive = FALSE) |>
    ln_p_change(log.p = log.p)
}



#' Get the predictability of each element of a vector of words (or phrases) using a causal transformer
#'
#' Get the predictability (by default the natural logarithm of the word probability) of each element of a vector of words (or phrases) given a
#' vector of left contexts using a using a causal transformer model. #'
#'
#' @param targets Target words.
#' @param l_contexts Left context for each word in `x`. If `l_contexts` is used,
#'        `by` is ignored. Set `by = NULL` to avoid a message notifying that.
#' @inheritParams causal_words_pred
#' @param ... not in use.
#' @inheritParams causal_preload
#' @inherit  causal_preload details
#' @inheritSection causal_next_tokens_pred_tbl More examples
#' @return A named vector of predictability values (by default the natural logarithm of the word probability).
#'
#' @examplesIf interactive()
#' causal_targets_pred(
#'   targets = c("tree.","cover."),
#'   l_contexts = c("The apple doesn't fall far from the",
#'                  "Don't judge a book by its"),
#'   model = "gpt2"
#' )
#' @family causal model functions
#' @export
causal_targets_pred <- function(targets,
                                l_contexts = NULL,
                                sep = " ",
                                log.p = getOption("pangoling.log.p"),
                                ignore_regex = "",
                                model = getOption("pangoling.causal.default"),
                                checkpoint = NULL,
                                add_special_tokens = NULL,
                                config_model = NULL,
                                config_tokenizer = NULL,
                                batch_size = 1,
                                ...) {
  dots <- list(...)
  # Check for unknown arguments
  if (length(dots) > 0) {
    unknown_args <- setdiff(names(dots), ".by")
    if (length(unknown_args) > 0) {
      stop("Unknown arguments: ", paste(unknown_args, collapse = ", "), ".")
    }
  }

  stride <- 1 # fixed for now
  message_verbose_model(model, checkpoint)
  x <- c(rbind(l_contexts, targets))
  by <- rep(seq_len(length(x)/2), each = 2)
  word_by_word_texts <- split(x, by, drop = TRUE)
  
  pasted_texts <- conc_words(word_by_word_texts, sep = sep)
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer
                    )
  trf <- lang_model(model,
                    checkpoint = checkpoint,
                    task = "causal",
                    config_model = config_model
                    )
  tensors <- create_tensor_lst(
    texts = unname(pasted_texts),
    tkzr = tkzr,
    add_special_tokens = add_special_tokens,
    stride = stride,
    batch_size = batch_size
  )

  lmats <- lapply(tensors, function(tensor) {
    causal_mat(tensor,
               trf,
               tkzr,
               add_special_tokens = add_special_tokens,
               stride = stride
               )
  }) |>
    unlist(recursive = FALSE)
  out <- tidytable::pmap(
                      list(
                        word_by_word_texts,
                        names(word_by_word_texts),
                        lmats
                      ),
                      function(words, item, mat) {
                        message_verbose(
                          "Text id: ", item, "\n`",
                          paste(words, collapse = sep),
                          "`"
                        )
                        word_lp(words,
                                sep = sep,
                                mat = mat,
                                ignore_regex = ignore_regex,
                                model = model,
                                add_special_tokens = add_special_tokens,
                                config_tokenizer = config_tokenizer
                                )
                      }
                    )

  keep <- c(FALSE, TRUE)

  out <- out |> lapply(function(x) x[keep])
  lps <- out |> unsplit(by[keep], drop = TRUE)

  names(lps) <- out |> lapply(function(x) paste0(names(x),"")) |>
    unsplit(by[keep], drop = TRUE)
  lps |>
    ln_p_change(log.p = log.p)
}
