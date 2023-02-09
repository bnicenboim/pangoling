#' Preloads a masked language model
#'
#' Preloads a masked language model to speed up next runs.
#'
#' A masked language model (also called BERT-like, or encoder model) is a type of large language model  that can be used to predict the content of a mask in a sentence.
#'
#' If not specified, the causal model that will be used is the one set in specified in the global option `pangoling.masked.default`, this can be accessed via `getOption("pangoling.masked.default")` (by default "`r getOption("pangoling.masked.default")`"). To change the default option use `options(pangoling.masked.default = "newmaskedmodel")`.
#'
#' A list of possible causal masked can be found in [hugging face website](https://huggingface.co/).
#'
#' Using the  `config_model` and `config_tokenizer` arguments, it's possible to control how the model and tokenizer from hugging face is accessed, see the python method [`from_pretrained`](https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained) for details. In case of errors check the status of [https://status.huggingface.co/](https://status.huggingface.co/)
#'
#' @inheritParams causal_preload
#' @return Nothing.
#'
#' @examplesIf interactive()
#' causal_preload(model = "bert-base-uncased")
#'
#' @family masked model functions
#' @export
#'
masked_preload <- function(model = getOption("pangoling.masked.default"),
                           add_special_tokens = NULL,
                           config_model = NULL, config_tokenizer = NULL) {
  lang_model(model, task = "masked", config_model)
  tokenizer(model, add_special_tokens = add_special_tokens, config_tokenizer)
  invisible()
}


#' Returns the configuration of a masked model
#'
#' Returns the configuration of a masked model.
#'
#' @inheritParams masked_preload
#' @inherit  masked_preload details
#' @return A list with the configuration of the model.
#' @examplesIf interactive()
#' masked_config(model = "bert-base-uncased")
#'
#' @family masked model functions
#' @export
masked_config <- function(model = getOption("pangoling.masked.default"),
                          config_model = NULL) {
  lang_model(model = model,
             task = "masked",
             config_model = config_model)$config$to_dict()
}

#' Get the possible tokens and their log probabilities for each mask in a sentence.
#'
#' For each mask in a sentence, get the possible tokens and their log probabilities using a masked transformer
#'
#' @section More examples:
#' See the  [online article](https://bruno.nicenboim.me/pangoling/articles/intro-bert.html) in pangoling website for more examples.
#'
#'
#' @param masked_sentences Masked sentences.
#' @inheritParams masked_preload
#' @inherit masked_preload details
#' @return A table with the masked sentences, the tokens (`token`), log probability (`lp`), and the respective mask number (`mask_n`).
#' @examplesIf interactive()
#' masked_tokens_tbl("The [MASK] doesn't fall far from the tree.",
#'                    model = "bert-base-uncased")
#'
#' @family masked model functions
#' @export
masked_tokens_tbl <- function(masked_sentences,
                                  model = getOption("pangoling.masked.default"),
                                  add_special_tokens = NULL,
                                  config_model = NULL,
                                  config_tokenizer = NULL) {
  message_verbose("Processing using masked model '", model, "'...")
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer)
  trf <- lang_model(model,
             task = "masked",
             config_model = config_model)
  vocab <- get_vocab(tkzr)
  # non_batched:
  tidytable::map_dfr(masked_sentences, function(masked_sentence) {
    masked_tensor <- encode(masked_sentence, tkzr,
                            add_special_tokens = add_special_tokens)
    outputs <- trf(masked_tensor)
    mask_pos <- which(masked_tensor$tolist()[[1]] == tkzr$mask_token_id)
    logits_masks <- outputs$logits[0][mask_pos - 1] # python starts in 0
    lp <- reticulate::py_to_r(torch$log_softmax(logits_masks, dim = -1L)$tolist())
    if (length(mask_pos) <= 1) lp <- list(lp) # to keep it consistent
    # names(lp) <-  1:length(lp)
    if (length(mask_pos) == 0) {
      tidytable::tidytable(masked_sentence = masked_sentence,
                           token = NA,
                           lp = NA,
                           mask_n = NA)
    } else {
      lp |> tidytable::map_dfr.(~
        tidytable::tidytable(
          masked_sentence = masked_sentence,
          token = vocab, lp = .x
        ) |>
          tidytable::arrange.(-lp), .id = "mask_n")
    }
  }) |>
    tidytable::relocate(mask_n, .after = tidyselect::everything())
}

#' Get the log probability of the last word (or phrase) of given a context
#'
#' Get the log probability of the last word (or phrase) of given a context using a masked transformer
#'
#' @section More examples:
#' See the  [online article](https://bruno.nicenboim.me/pangoling/articles/intro-bert.html) in pangoling website for more examples.
#'
#'
#' @param contexts Context sentences.
#' @param last_words One last word for each context sentence
#' @param final_punctuation Punctuation at the end of the sentence
#' @inheritParams masked_preload
#' @inheritParams causal_lp
#' @inherit masked_preload details
#' @return A named vector of log probabilities.
#' @examplesIf interactive()
#' masked_last_lp(c("The apple doesn't fall far from the",
#'                   "The tree doesn't fall far from the"),
#'                last_words = c("tree","apple"),
#'                model = "bert-base-uncased")
#'
#' @family masked model functions
#' @export
masked_last_lp <- function(contexts,
                           last_words,
                           ignore_regex = "",
                           final_punctuation = ".",
                           model = getOption("pangoling.masked.default"),
                           add_special_tokens = NULL,
                           config_model = NULL,
                           config_tokenizer = NULL) {
  stride <- 1
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer)
  trf <- lang_model(model,
                    task = "masked",
                    config_model = config_model)

  message_verbose("Processing using masked model '", model, "'...")

  # word_by_word_texts <- get_word_by_word_texts(x, .by)
  last_tokens <- char_to_token(last_words,tkzr)
  masked_sentences <- tidytable::map2_chr(contexts, last_tokens, ~ {
    paste0(.x,
           " ",
           paste0(rep(tkzr$mask_token, length(.y)), collapse = ""),
           final_punctuation)
  })

  # named tensor list:
  tensors_lst <- tidytable::map2(masked_sentences, last_words, function(t, w) {
    l <- create_tensor_lst(t,
                           tkzr,
                           add_special_tokens = add_special_tokens,
                           stride = stride)
    names(l) <- w
    l
  })

  out <- tidytable::pmap.(list(last_words, contexts, tensors_lst),
                          function(words, item, tensor_lst) {
    # words <- word_by_word_texts[[1]]
    # item <- names(word_by_word_texts[[1]])
    # tensor_lst <- tensors_lst[[1]]
    ls_mat <- masked_lp_mat(tensor_lst,
                            trf = trf,
                            tkzr = tkzr,
                            add_special_tokens = add_special_tokens,
                            stride = stride)
    text <- paste0(words, collapse = " ")
    tokens <- char_to_token(text, tkzr)[[1]]
    lapply(ls_mat, function(m) {
      # m <- ls_mat[[1]]
      message_verbose("Context: ", item, "\n`", paste(words, collapse = " "), "`")
      word_lp(words,
              mat = m,
              ignore_regex = ignore_regex,
              model = model,
              add_special_tokens = add_special_tokens,
              config_tokenizer = config_tokenizer)
    })
    # out_ <- lapply(1:length(out[[1]]), function(i) lapply(out, "[", i))
  })
  unlist(out, recursive = TRUE)
}


#' @noRd
masked_lp_mat <- function(tensor_lst,
                          trf,
                          tkzr,
                          add_special_tokens = NULL,
                          stride = 1,
                          N_pred =1) {

  tensor <- torch$row_stack(unname(tensor_lst))
  words <- names(tensor_lst)
  tokens <- char_to_token(words, tkzr)
  n_masks <- sum(tensor_lst[[1]]$tolist()[[1]] == tkzr$mask_token_id)
  message_verbose("Processing ", tensor$shape[0], " batch(es) of ", tensor$shape[1], " tokens.")

  out_lm <- trf(tensor)
  logits_b <- out_lm$logits

  is_masked_lst <- lapply(tensor_lst, function(t) {
    # t <- tensor_lst[[1]]
    id_vector <- t$tolist()[[1]]
    id_vector %in% tkzr$mask_token_id
  })
  # number of predictions ahead
  # if(is.null(N_pred)) N_pred <- sum(is_masked_lst[[1]])
  if (is.null(N_pred)) N_pred <- length(words)

  lmat <- lapply(1:N_pred, function(n_pred) {
    logits_masked <- lapply(seq_along(tensor_lst), function(n) {
      # n <- 1
      # logits is a python object indexed from 0
      if ((n - n_pred) < 0) {
        return(NULL)
      }
      n_masks_here <- length(tokens[[n]])
      n_pred_element <- which(is_masked_lst[[n]])[1:n_masks_here]
      # if(!is_masked_lst[[n]][n_pred_element] #outside of masked elements
      #    || anyNA(n_pred_element)) {
      #   return(NULL)
      # }
      # iterates over sentences
      logits_b[n - n_pred][n_pred_element - 1]
    })
    logits_masked_cleaned <-
      logits_masked[lengths(logits_masked) > 0] |>
      torch$row_stack()
    lp <- reticulate::py_to_r(torch$log_softmax(logits_masked_cleaned,
      dim = -1L
    ))$tolist()
    mat <- do.call("cbind", lp)
    # columns are not named
    mat_NA <- matrix(NA,
      nrow = nrow(mat),
      ncol = sum(lengths(logits_masked) == 0)
    )
    # add NA columns for predictions not made
    mat <- cbind(mat_NA, mat)
    colnames(mat) <- unlist(tokens)
    rownames(mat) <- get_vocab(tkzr)
    mat
  })
  gc(full = TRUE)
  lmat
}
