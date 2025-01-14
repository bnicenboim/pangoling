#' Preloads a causal language model
#'
#' Preloads a causal language model to speed up next runs.
#' 
#' @section More details about causal models:
#' A causal language model (also called GPT-like, auto-regressive, or decoder
#' model) is a type of large language model usually used for text-generation
#' that can predict the next word (or more accurately in fact token) based
#' on a preceding context.
#'
#' If not specified, the causal model used will be the one set in the global
#' option `pangoling.causal.default`, this can be
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
#' @param model Name of a pre-trained model or folder. One should be able to use
#' models based on "gpt2". See 
#' [hugging face website](https://huggingface.co/models?other=gpt2).
#' @param checkpoint Folder of a checkpoint.
#' @param add_special_tokens Whether to include special tokens. It has the
#'                           same default as the
#' [AutoTokenizer](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoTokenizer)
#'                            method in Python.
#' @param config_model List with other arguments that control how the
#'                      model from Hugging Face is accessed.
#' @param config_tokenizer List with other arguments that control how the 
#'                         tokenizer from Hugging Face is accessed.
#'
#' @return Nothing.
#'
#' @examples
#' causal_preload(model = "gpt2")
#'
#' @family causal model helper functions
#' @export
#'
causal_preload <- function(model = getOption("pangoling.causal.default"),
                           checkpoint = NULL,
                           add_special_tokens = NULL,
                           config_model = NULL, config_tokenizer = NULL) {
  message_verbose("Preloading causal model ", model, "...")
  lang_model(model, 
             checkpoint = checkpoint, 
             task = "causal", 
             config_model = config_model)
  tokenizer(model, 
            add_special_tokens = add_special_tokens, 
            config_tokenizer = config_tokenizer)
  invisible()
}

#' Returns the configuration of a causal model
#'
#' @inheritParams causal_preload
#' @inherit  causal_preload details
#' @inheritSection causal_preload More details about causal models
#' @return A list with the configuration of the model.
#' @examples
#' causal_config(model = "gpt2")
#'
#' @family causal model helper functions
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

#' Get the possible next tokens and their predictability based on the
#' previous context using a causal transformer
#'
#' Get the possible next tokens and their predictabilities based on a
#' previous context using a causal transformer model from 
#' [Hugging Face](https://huggingface.co).
#'
#' @section More examples:
#' See the
#' [online article](https://bruno.nicenboim.me/pangoling/articles/intro-gpt2.html)
#' on the pangoling website for more examples.
#'
#' @param context The context.
#' @param decode Should it decode the tokens into readable strings? This is 
#'               relevant for special characters such as accents and 
#'               diacritics, which get mangled in the tokens.
#' @inheritParams causal_preload
#' @inheritParams causal_tokens_pred_lst
#' @inherit  causal_preload details
#' @return A table with possible next tokens and their log-probabilities.
#' @examples
#' causal_next_tokens_pred_tbl(
#'   context = "The apple doesn't fall far from the",
#'   model = "gpt2"
#' )
#'
#' @family causal model functions
#' @export
causal_next_tokens_pred_tbl <- 
  function(context,
           log.p = getOption("pangoling.log.p"),
           decode = FALSE,
           model = getOption("pangoling.causal.default"),
           checkpoint = NULL,
           add_special_tokens = NULL,
           config_model = NULL,
           config_tokenizer = NULL) {
    if (length(unlist(context)) > 1) {
      stop2("Only one context is allowed in this function.")
    }
    if(any(!is_really_string(context))) {
      stop2("`context` needs to contain a string.")
    }
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
    context_tensor <- encode(list(unlist(context)),
                             tkzr,
                             add_special_tokens = add_special_tokens
                             )$input_ids
    generated_outputs <- trf(context_tensor)
    n_tokens <- length(context_tensor$tolist()[[1]]) # was 0
    logits_next_word <- generated_outputs$logits[0][n_tokens - 1]
    l_softmax <- torch$log_softmax(logits_next_word, dim = -1L)$tolist()
    lp <- reticulate::py_to_r(l_softmax) |>
      unlist()
    vocab <- get_vocab(tkzr, decode = decode)
    diff_words <- length(vocab) - length(lp)
    if(diff_words > 0) {
      warning(paste0("Tokenizer's vocabulary is longer than the model's.",
                     " Some words will have NA predictability."))
      lp <- c(lp, rep(NA, diff_words))
    } else if(diff_words < 0) {
      stop2("Tokenizer's vocabulary is smaller than the model's.")
    }
    tidytable::tidytable(token = vocab,
                         pred = lp |> ln_p_change(log.p = log.p)) |>
      tidytable::arrange(-pred)
  }



#' Compute predictability using a causal transformer model
#'
#' These functions calculate the predictability of words, phrases, or tokens 
#' using a causal transformer model. 
#'
#' @details
#' These functions calculate the predictability (by default the natural 
#' logarithm of the word probability) of words, phrases or tokens using a causal 
#' transformer model:
#' 
#' - **`causal_targets_pred()`**: Evaluates specific target words or phrases 
#'   based on their given contexts. Use when you have explicit
#'   context-target pairs to evaluate, with each target word or phrase paired 
#'   with a single preceding context.
#' - **`causal_words_pred()`**: Computes predictability for all elements of a 
#'   vector grouped by a specified variable. Use when working with words or 
#'   phrases split into groups, such as sentences or paragraphs, where 
#'   predictability is computed for every word or phrase in each group.
#' - **`causal_tokens_pred_lst()`**: Computes the predictability of each token 
#'   in a sentence (or group of sentences) and returns a list of results for 
#'   each sentence. Use when you want to calculate the predictability of 
#'   every token in one or more sentences.
#'
#' See the
#' [online article](https://bruno.nicenboim.me/pangoling/articles/intro-gpt2.html)
#' in pangoling website for more examples.
#' 
#' @param targets A character vector of target words or phrases (for 
#'                `causal_targets_pred()`).
#' @param contexts A character vector of contexts corresponding to each target 
#'                 (for `causal_targets_pred()`).
#' @param x A character vector of words, phrases, or texts to evaluate (for 
#'          `causal_words_pred()`).
#' @param by A grouping variable indicating how texts are split into groups (for
#'          `causal_words_pred()`).
#' @param sep A string specifying how words are separated within contexts or 
#'            groups. Default is `" "`. For languages that don't have spaces 
#'            between words (e.g., Chinese), set `sep = ""`.
#' @param texts A vector or list of sentences or paragraphs (for
#'              `causal_tokens_pred_lst()`).
#' @param log.p Base of the logarithm used for the output predictability values.
#'              If `TRUE` (default), the natural logarithm (base *e*) is used.
#'              If `FALSE`, the raw probabilities are returned.
#'              Alternatively, `log.p` can be set to a numeric value specifying
#'              the base of the logarithm (e.g., `2` for base-2 logarithms).
#'              To get surprisal in bits (rather than predictability), set
#'              `log.p = 1/2`.
#' @param ignore_regex Can ignore certain characters when calculating the log
#'                      probabilities. For example `^[[:punct:]]$` will ignore
#'                      all punctuation  that stands alone in a token.
#' @param batch_size Maximum size of the batch. Larger batches speed up
#'                   processing but take more memory.
#' @inheritParams causal_preload
#' @inheritSection causal_preload More details about causal models
#' @inheritSection causal_next_tokens_pred_tbl More examples
#' @param ... Currently not in use.
#' @return For `causal_targets_pred()` and `causal_words_pred()`, 
#'   a named numeric vector of predictability scores. For 
#'   `causal_tokens_pred_lst()`, a list of named numeric vectors, one for 
#'   each sentence or group.
#'
#' @examples
#' # Using causal_targets_pred
#' causal_targets_pred(
#'   targets = c("tree.", "cover."),
#'   contexts = c("The apple doesn't fall far from the",
#'                "Don't judge a book by its"),
#'   model = "gpt2"
#' )
#'
#' # Using causal_words_pred
#' causal_words_pred(
#'   x = df_sent$word,
#'   by = df_sent$sent_n,
#'   model = "gpt2"
#' )
#' 
#' # Using causal_tokens_pred_lst
#' preds <- causal_tokens_pred_lst(
#'   texts = c("The apple doesn't fall far from the tree.",
#'             "Don't judge a book by its cover."),
#'   model = "gpt2"
#' )
#' # Convert the output to a tidy table
#' library(tidytable)
#' map2_dfr(preds, seq_along(preds), 
#' ~ data.frame(tokens = names(.x), pred = .x, id = .y))
#'
#' @family causal model functions
#' @export
#' @rdname causal_predictability
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
  if(any(!is_really_string(x))) {
    stop2("`x` needs to be a vector of non-empty strings.")
  }
  dots <- list(...)
  # Check for unknown arguments
  if (length(dots) > 0) {
    unknown_args <- setdiff(names(dots), ".by")
    if (length(unknown_args) > 0) {
      stop("Unknown arguments: ", paste(unknown_args, collapse = ", "), ".")
    }
  }
  if (length(x) != length(by)) {
    stop2("The argument `by` has an incorrect length.")
  }

  if(any(x != trimws(x)) & sep == " ") {
    message_verbose(paste0("Notice that some words have white spaces,",
                           ' argument `sep` should probably set to "".'))
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
               decode = FALSE,
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

  message_verbose("***\n")
  lps <- out |> unsplit(by, drop = TRUE)
  names(lps) <- out |> lapply(function(x) paste0(names(x),"")) |>
    unsplit(by, drop = TRUE)
  lps |> ln_p_change(log.p = log.p)
}




#' @rdname causal_predictability
#' @export
causal_tokens_pred_lst <- 
  function(texts,
           log.p = getOption("pangoling.log.p"),
           model = getOption("pangoling.causal.default"),
           checkpoint = NULL,
           add_special_tokens = NULL,
           config_model = NULL,
           config_tokenizer = NULL,
           batch_size = 1) {
    if(any(!is_really_string(texts))){
      stop2("`texts` needs to be a vector of non-empty strings.")
    }
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
                 decode = FALSE,
                 stride = stride)
    }) |>
      unlist(recursive = FALSE)

    lapply(ls_mat, function(mat) {
      if (ncol(mat) == 1 && colnames(mat) == "") {
        
                     pred = NA_real_
                   names(pred) =""
      } else {
               pred = tidytable::map2_dbl(colnames(mat),
                                                seq_len(ncol(mat)),
                                                ~ mat[.x, .y]) |>
                       ln_p_change(log.p = log.p)
               names(pred) = colnames(mat)
               
      }
      pred
    })
  }


#' @noRd
causal_mat <- function(tensor,
                       trf,
                       tkzr,
                       add_special_tokens = NULL,
                       decode,
                       stride = 1) {
  message_verbose(
    "Processing a batch of size ",
    tensor$input_ids$shape[0],
    " with ",
    tensor$input_ids$shape[1], " tokens."
  )

  if (tensor$input_ids$shape[1] == 0) {
    warning("No tokens found.", call. = FALSE)
    vocab <- get_vocab(tkzr, decode = decode)
    mat <- matrix(rep(NA, length(vocab)), ncol = 1)
    rownames(mat) <- vocab
    colnames(mat) <- ""
    return(list(mat))
  }

  logits_b <- trf$forward(
                    input_ids = tensor$input_ids,
                    attention_mask = tensor$attention_mask
                  )$logits

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
    # in case the last words in the vocab were not used to train the model
    vocab <- get_vocab(tkzr, decode = decode)
    diff_words <- length(vocab) - nrow(mat)
    if(diff_words > 0) {
      warning("Tokenizer's vocabulary is larger than the model's.")
    } else if(diff_words < 0) {
      stop2("Tokenizer's vocabulary is smaller than the model's.")
    }
    rownames(mat) <- vocab[seq_len(nrow(mat))]
    colnames(mat) <- unlist(tokens)
    mat
  })
  rm(logits_b)
  lmat
}



#' Generate a list of predictability matrices using a causal transformer model
#'
#' This function computes a list of matrices, where each matrix corresponds to a
#' unique group specified by the `by` argument. Each matrix represents the
#' predictability of every token in the input text (`x`) based on preceding 
#' context, as evaluated by a causal transformer model.
#'
#'
#' @details
#' The function splits the input `x` into groups specified by the `by` argument 
#' and processes each group independently. For each group, the model computes
#' the predictability of each token in its vocabulary based on preceding context.
#'
#' Each matrix contains:
#' - Rows representing the model's vocabulary.
#' - Columns corresponding to tokens in the group (e.g., a sentence or
#' paragraph).
#' - By default, values are  the natural logarithm of word probabilities.
#'
#' @inheritParams causal_words_pred
#' @inheritParams causal_preload
#' @inheritParams causal_next_tokens_pred_tbl
#' @param sorted When default FALSE it will retain the order of groups we are 
#'               splitting by. When TRUE then sorted (according to `by`) list(s)
#'               are returned. 
#' @inherit  causal_preload details
#' @inheritSection causal_preload More details about causal models
#' @inheritSection causal_next_tokens_pred_tbl More examples
#' @return A list of matrices with tokens in their columns and the vocabulary of
#'         the model in their rows
#'
#' @examples
#' list_of_mats <- causal_pred_mats(
#'                        x = df_sent$word,
#'                        by = df_sent$sent_n,  
#'                        model = "gpt2"
#'                 )
#' list_of_mats |> str()
#' list_of_mats[[1]] |> tail()
#' list_of_mats[[2]] |> tail()
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
                             decode = FALSE,
                             config_model = NULL,
                             config_tokenizer = NULL,
                             batch_size = 1,
                             ...) {
  if(any(!is_really_string(x))) {
    stop2("`x` needs to be a vector of non-empty strings.")
  }
  dots <- list(...)
  if(any(x != trimws(x)) & sep == " ") {
    message_verbose(paste0('Notice that some words have white spaces,',
                           ' argument `sep` should probably set to "".'))
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
                                    trf = trf,
                                    tkzr = tkzr,
                                    add_special_tokens = add_special_tokens,

                                    decode = decode,
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


#' @export
#' @rdname causal_predictability
causal_targets_pred <- function(targets,
                                contexts = NULL,
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
  if(any(!is_really_string(targets))) { 
    stop2("`targets` needs to be a vector of non-empty strings.")
  }
  if(any(!is_really_string(contexts))) {
    stop2("`contexts` needs to be a vector of non-empty strings.")
  }
  dots <- list(...)
  # Check for unknown arguments
  if (length(dots) > 0) {
    unknown_args <- setdiff(names(dots), ".by")
    if (length(unknown_args) > 0) {
      stop("Unknown arguments: ", paste(unknown_args, collapse = ", "), ".")
    }
  }
  if(any(targets != trimws(targets)) | 
     any(contexts != trimws(contexts)) & sep == " ") {
    message_verbose(
      paste0('Notice that some words have white spaces,',
             ' if this is intended, argument `sep` should probably set to "".'))
  }
  stride <- 1 # fixed for now
  message_verbose_model(model, checkpoint)
  x <- c(rbind(contexts, targets))
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
               trf = trf,
               tkzr = tkzr,
               add_special_tokens = add_special_tokens,
               decode = FALSE,
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
  message_verbose("***\n")
  keep <- c(FALSE, TRUE)

  out <- out |> lapply(function(x) x[keep])
  lps <- out |> unsplit(by[keep], drop = TRUE)

  names(lps) <- out |> lapply(function(x) paste0(names(x),"")) |>
    unsplit(by[keep], drop = TRUE)
  lps |>
    ln_p_change(log.p = log.p)
}
