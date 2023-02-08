#' get_masked_tokens_tbl
#'
#' @param context Context
#' @param model Name of a pretrained model stored on the huggingface.co. (Maybe a path to a  model (.pt or .bin file) stored locally will work.)
#'
#' @return A table
#' @export
get_masked_tokens_tbl <- function(masked_sentences,
                                  model = "bert-base-uncased",
                                  add_special_tokens = NULL,
                                  config_model = NULL,
                                  config_tokenizer = NULL) {
  message_verbose("Processing using masked model '", model, "'...")
  tkzr <- tokenizer(model, add_special_tokens = add_special_tokens, config = config_tokenizer)
  vocab <- get_vocab(tkzr)
  if (!is.null(add_special_tokens)) {
    encode <- function(x) tkzr$encode(x, return_tensors = "pt", add_special_tokens = add_special_tokens)
  } else {
    encode <- function(x) tkzr$encode(x, return_tensors = "pt")
  }
  # non_batched:
  tidytable::map_dfr(masked_sentences, function(masked_sentence) {
    masked_tensor <- encode(masked_sentence)
    outputs <- lang_model(model, task = "masked", config = config_model)(masked_tensor)
    mask_pos <- which(masked_tensor$tolist()[[1]] == tkzr$mask_token_id)

    logits_masks <- outputs$logits[0][mask_pos - 1] # python starts in 0
    lp <- reticulate::py_to_r(torch$log_softmax(logits_masks, dim = -1L)$tolist())
    if (length(mask_pos) <= 1) lp <- list(lp) # to keep it consistent
    # names(lp) <-  1:length(lp)
    if (length(mask_pos) == 0) {
      tidytable::tidytable(masked_sentence = masked_sentence, token = NA, lp = NA, mask_n = NA)
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

#' @export
get_masked_last_lp <- function(contexts, last_words, ignore_regex = "", final_punctuation = ".", model = "bert-base-uncased", add_special_tokens = NULL, config_model = NULL, config_tokenizer = NULL) {
  tkzr <- tokenizer(model, add_special_tokens = add_special_tokens, config = config_tokenizer)
  message_verbose("Processing using masked model '", model, "'...")

  # word_by_word_texts <- get_word_by_word_texts(x, .by)
  last_tokens <- get_tokens(last_words, model = model, add_special_tokens = add_special_tokens, config = config_tokenizer)
  masked_sentences <- tidytable::map2_chr(contexts, last_tokens, ~ {
    paste0(.x, " ", paste0(rep(tkzr$mask_token, length(.y)), collapse = ""), final_punctuation)
  })

  # named tensor list:
  tensors_lst <- tidytable::map2(masked_sentences, last_words, function(t, w) {
    l <- create_tensor_lst(t, model = model, add_special_tokens = add_special_tokens, stride = stride, config = config_tokenizer)
    names(l) <- w
    l
  })

  out <- tidytable::pmap.(list(last_words, contexts, tensors_lst), function(words, item, tensor_lst) {
    # words <- word_by_word_texts[[1]]
    # item <- names(word_by_word_texts[[1]])
    # tensor_lst <- tensors_lst[[1]]
    ls_mat <- masked_lp_mat(tensor_lst, model = model, add_special_tokens = add_special_tokens, stride = stride, config_model = config_model, config_tokenizer = config_tokenizer, N_pred = 1)
    text <- paste0(words, collapse = " ")
    tokens <- get_tokens(text, model = model, add_special_tokens = add_special_tokens, config = config_tokenizer)[[1]]
    lapply(ls_mat, function(m) {
      # m <- ls_mat[[1]]
      message_verbose("Context: ", item, "\n`", paste(words, collapse = " "), "`")
      word_lp(words, mat = m, ignore_regex = ignore_regex, model = model, add_special_tokens = add_special_tokens, config_tokenizer = config_tokenizer)
    })
    # out_ <- lapply(1:length(out[[1]]), function(i) lapply(out, "[", i))
  })
  unlist(out, recursive = TRUE)
}

#' Get the log probability of each word phrase of a vector given its previous context using a transformer model from huggingface.co.
#'
#' In case of errors check the status of https://status.huggingface.co/
#'
#' @param x Vector of words, phrases or texts.
#' @param .by Vector that indicates how the text should be split.
#' @param model Name of a pretrained model stored on the huggingface.co. (Maybe a path to a  model (.pt or .bin file) stored locally will work.)
#' @ignore_regex Can ignore certain characters when calculates the log probabilities. For example `^[[:punct:]]$` will ignore all punctuation  that stands alone in a token.
#'
#' @return a vector of log probabilities.
#' @export
get_masked_lp <- function(x, .by = rep(1, length(x)), ignore_regex = "", model = "bert-base-uncased", add_special_tokens = NULL, stride = 1, config_model = NULL, config_tokenizer = NULL) {
  tkzr <- tokenizer(model, add_special_tokens = add_special_tokens, config = config_tokenizer)
  message_verbose("Processing using masked model '", model, "'...")

  word_by_word_texts <- get_word_by_word_texts(x, .by)

  masked_word_by_word_texts <- lapply(word_by_word_texts, function(word_by_word_text) {
    # word_by_word_text <- word_by_word_texts[[1]]
    len <- length(word_by_word_text)
    tokens <- get_tokens(word_by_word_text, model = model, add_special_tokens = add_special_tokens, config = config_tokenizer)
    lapply(1:len, function(pos) {
      word_by_word_text[pos:len] <- tidytable::map_chr(lengths(tokens[pos:len]), ~ paste0(rep(tkzr$mask_token, .x), collapse = ""))
      word_by_word_text
    })
  })

  # N <- length(word_by_word_texts)
  pasted_masked_texts <- lapply(masked_word_by_word_texts, function(t) lapply(t, function(word) paste0(word, collapse = " ")))

  # named tensor list:
  tensors_lst <- tidytable::map2(pasted_masked_texts, word_by_word_texts, function(t, w) {
    l <- create_tensor_lst(t, model = model, add_special_tokens = add_special_tokens, stride = stride, config = config_tokenizer)
    names(l) <- w
    l
  })

  out <- tidytable::pmap.(list(word_by_word_texts, names(word_by_word_texts), tensors_lst), function(words, item, tensor_lst) {
    # words <- word_by_word_texts[[1]]
    # item <- names(word_by_word_texts[[1]])
    # tensor_lst <- tensors_lst[[1]]
    ls_mat <- masked_lp_mat(tensor_lst, model = model, add_special_tokens = add_special_tokens, stride = stride, config_model = config_model, config_tokenizer = config_tokenizer, N_pred = 1)
    text <- paste0(words, collapse = " ")
    tokens <- get_tokens(text, model = model, add_special_tokens = add_special_tokens, config = config_tokenizer)[[1]]
    lapply(ls_mat, function(m) {
      # m <- ls_mat[[1]]
      message_verbose("Text id: ", item, "\n`", paste(words, collapse = " "), "`")
      word_lp(words, mat = m, ignore_regex = ignore_regex, model = model, add_special_tokens = add_special_tokens, config_tokenizer = config_tokenizer)
    })
    # out_ <- lapply(1:length(out[[1]]), function(i) lapply(out, "[", i))
  })
  unlist(out, recursive = TRUE)
}



#' @noRd
masked_lp_mat <- function(tensor_lst, model = "bert-base-uncased", add_special_tokens = NULL, stride = 1, config_model = NULL, config_tokenizer = NULL, N_pred = NULL) {
  tkzr <- tokenizer(model, add_special_tokens = add_special_tokens, config = config_tokenizer)

  tensor <- torch$row_stack(unname(tensor_lst))
  words <- names(tensor_lst)
  tokens <- get_tokens(words, model, add_special_tokens = add_special_tokens, config = config_tokenizer)
  n_masks <- sum(tensor_lst[[1]]$tolist()[[1]] == tkzr$mask_token_id)
  message_verbose("Processing ", tensor$shape[0], " batch(es) of ", tensor$shape[1], " tokens.")


  out_lm <- lang_model(model, task = "masked", config = config_model)(tensor)
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
    rownames(mat) <- get_tr_vocab(model, add_special_tokens = add_special_tokens, config = config_tokenizer)
    mat
  })
  gc(full = TRUE)
  lmat
}
