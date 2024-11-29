#' Get the possible tokens and their log probabilities for each mask in a sentence
#'
#' For each mask in a sentence, get the possible tokens and their log
#' probabilities using a masked transformer.
#'
#' @section More examples:
#' See the
#' [online article](https://bruno.nicenboim.me/pangoling/articles/intro-bert.html)
#' in pangoling website for more examples.
#'
#'
#' @param masked_sentences Masked sentences.
#' @inheritParams masked_preload
#' @inherit masked_preload details
#' @return A table with the masked sentences, the tokens (`token`),
#'         log probability (`lp`), and the respective mask number (`mask_n`).
#' @examplesIf interactive()
#' masked_tokens_tbl("The [MASK] doesn't fall far from the tree.",
#'   model = "bert-base-uncased"
#' )
#'
#' @family masked model functions
#' @export
masked_tokens_tbl <- function(masked_sentences,
                              model = getOption("pangoling.masked.default"),
                              add_special_tokens = NULL,
                              config_model = NULL,
                              config_tokenizer = NULL) {
 .Deprecated(new = "masked_tokens_pred_tbl()")
 message_verbose("Processing using masked model '", model, "'...")
  tkzr <- tokenizer(model,
    add_special_tokens = add_special_tokens,
    config_tokenizer = config_tokenizer
  )
  trf <- lang_model(model,
    task = "masked",
    config_model = config_model
  )
  vocab <- get_vocab(tkzr)
  # non_batched:
  # TODO: speedup using batches
  tidytable::map_dfr(masked_sentences, function(masked_sentence) {
    masked_tensor <- encode(list(masked_sentence), tkzr,
      add_special_tokens = add_special_tokens
    )$input_ids
    outputs <- trf(masked_tensor)
    mask_pos <- which(masked_tensor$tolist()[[1]] == tkzr$mask_token_id)
    logits_masks <- outputs$logits[0][mask_pos - 1] # python starts in 0
    lp <- reticulate::py_to_r(
      torch$log_softmax(logits_masks, dim = -1L)$tolist()
    )
    if (length(mask_pos) <= 1) lp <- list(lp) # to keep it consistent
    # names(lp) <-  1:length(lp)
    if (length(mask_pos) == 0) {
      tidytable::tidytable(
        masked_sentence = masked_sentence,
        token = NA,
        lp = NA,
        mask_n = NA
      )
    } else {
      lp |> tidytable::map_dfr(~
        tidytable::tidytable(
          masked_sentence = masked_sentence,
          token = vocab, lp = .x
        ) |>
          tidytable::arrange(-lp), .id = "mask_n")
    }
  }) |>
    tidytable::relocate(mask_n, .after = tidyselect::everything())
}

#' Get the log probability of a target word (or phrase) given a left and right context
#'
#' Get the log probability of a vector of target words (or phrase) given a
#' vector of left and of right contexts using a masked transformer.
#'
#' @section More examples:
#' See the
#' [online article](https://bruno.nicenboim.me/pangoling/articles/intro-bert.html)
#' in pangoling website for more examples.
#'
#'
#' @param l_contexts Left context of the target word.
#' @param targets Target words.
#' @param r_contexts Right context of the target word.
#' @inheritParams masked_preload
#' @inheritParams causal_lp
#' @inherit masked_preload details
#' @return A named vector of log probabilities.
#' @examplesIf interactive()
#' masked_lp(
#'   l_contexts = c("The", "The"),
#'   targets = c("apple", "pear"),
#'   r_contexts = c(
#'     "doesn't fall far from the tree.",
#'     "doesn't fall far from the tree."
#'   ),
#'   model = "bert-base-uncased"
#' )
#'
#' @family masked model functions
#' @export
masked_lp <- function(l_contexts,
                      targets,
                      r_contexts,
                      ignore_regex = "",
                      model = getOption("pangoling.masked.default"),
                      add_special_tokens = NULL,
                      config_model = NULL,
                      config_tokenizer = NULL) {
 .Deprecated(new = "masked_targets_pred()")
  stride <- 1
  tkzr <- tokenizer(model,
    add_special_tokens = add_special_tokens,
    config_tokenizer = config_tokenizer
  )
  trf <- lang_model(model,
    task = "masked",
    config_model = config_model
  )

  message_verbose("Processing using masked model '", model, "'...")

  target_tokens <- char_to_token(targets, tkzr)
  masked_sentences <- tidytable::pmap_chr(
    list(
      l_contexts,
      target_tokens,
      r_contexts
    ),
    function(l, target, r) {
      paste0(
        l,
        " ",
        paste0(rep(tkzr$mask_token, length(target)), collapse = ""),
        " ",
        r
      )
    }
  )

  # named tensor list:
  tensors_lst <- tidytable::map2(masked_sentences, targets, function(t, w) {
    l <- create_tensor_lst(t,
      tkzr,
      add_special_tokens = add_special_tokens,
      stride = stride
    )
    names(l) <- w
    l
  })

  out <- tidytable::pmap(
    list(targets, l_contexts, r_contexts, tensors_lst),
    function(words, l, r, tensor_lst) {
      # TODO: make it by batches
      ls_mat <- masked_lp_mat(lapply(tensor_lst, function(t) t$input_ids),
        trf = trf,
        tkzr = tkzr,
        add_special_tokens = add_special_tokens,
        stride = stride
      )
      text <- paste0(words, collapse = " ")
      tokens <- char_to_token(text, tkzr)[[1]]
      lapply(ls_mat, function(m) {
        # m <- ls_mat[[1]]
        message_verbose(l, " [", words, "] ", r)

        word_lp(words,
          mat = m,
          sep = " ",
          ignore_regex = ignore_regex,
          model = model,
          add_special_tokens = add_special_tokens,
          config_tokenizer = config_tokenizer
        )
      })
      # out_ <- lapply(1:length(out[[1]]), function(i) lapply(out, "[", i))
    }
  )
  unlist(out, recursive = TRUE)
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
causal_next_tokens_tbl <- function(context,
                                   model = getOption("pangoling.causal.default"),
                                   checkpoint = NULL,
                                   add_special_tokens = NULL,
                                   config_model = NULL,
                                   config_tokenizer = NULL) {

 .Deprecated(new = "causal_next_tokens_pred_tbl()")

  if (length(unlist(context)) > 1) stop2("Only one context is allowed in this function.")
  message_verbose("Processing using causal model '", file.path(model, checkpoint), "'...")
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
  n_tokens <- length(context_tensor$tolist()[0])
  logits_next_word <- generated_outputs$logits[0][n_tokens - 1]
  l_softmax <- torch$log_softmax(logits_next_word, dim = -1L)$tolist()
  lp <- reticulate::py_to_r(l_softmax) |>
    unlist()
  vocab <- get_vocab(tkzr)
  tidytable::tidytable(token = vocab, lp = lp) |>
    tidytable::arrange(-lp)
}


#' Get the log probability of each element of a vector of words (or phrases) using a causal transformer
#'
#' Get the log probability of each element of a vector of words (or phrases) using a causal transformer model. See the
#' [online article](https://bruno.nicenboim.me/pangoling/articles/intro-gpt2.html)
#' in pangoling website for more examples.
#'
#'
#' @param x Vector of words, phrases or texts.
#' @param by Vector that indicates how the text should be split.
#' @param l_contexts Left context for each word in `x`. If `l_contexts` is used,
#'        `by` is ignored. Set `by = NULL` to avoid a message notifying that.
#' @param ... not in use.
#' @inheritParams causal_preload
#' @param ignore_regex Can ignore certain characters when calculates the log
#'                      probabilities. For example `^[[:punct:]]$` will ignore
#'                      all punctuation  that stands alone in a token.
#' @param batch_size Maximum size of the batch. Larges batches speedup
#'                   processing but take more memory.
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
#'causal_lp(
#'   x = "tree.",
#'   l_contexts = "The apple doesn't fall far from the tree.",
#'   by = NULL, # it's ignored anyways
#'   model = "gpt2"
#' )

#' @family causal model functions
#' @export
causal_lp <- function(x,
                      by = rep(1, length(x)),
                      l_contexts = NULL,
                      ignore_regex = "",
                      model = getOption("pangoling.causal.default"),
                      checkpoint = NULL,
                      add_special_tokens = NULL,
                      config_model = NULL,
                      config_tokenizer = NULL,
                      batch_size = 1,
                      ...) {
 .Deprecated(new = "causal_targets_pred() supporting the l_context argument or causal_words_pred() for the x and by arguments.")
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

  stride <- 1 # fixed for now
  message_verbose("Processing using causal model '", file.path(model, checkpoint), "'...")
  if(!is.null(l_contexts)){
    if(all(!is.null(by))) message_verbose("Ignoring `by` argument")
    x <- c(rbind(l_contexts, x))
    by <- rep(seq_len(length(x)/2), each = 2)
  }
  word_by_word_texts <- split(x, by, drop = TRUE)
  
  pasted_texts <- lapply(
    word_by_word_texts,
    function(word) paste0(word, collapse = " ")
  )
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
      # words <- word_by_word_texts[[1]]
      # item <- names(word_by_word_texts)[[1]]
      # mat <- lmats[[1]]

      message_verbose(
        "Text id: ", item, "\n`",
        paste(words, collapse = " "),
        "`"
      )
      word_lp(words,
        mat = mat,
        sep = " ",
        ignore_regex = ignore_regex,
        model = model,
        add_special_tokens = add_special_tokens,
        config_tokenizer = config_tokenizer
      )
    }
  )
  if(!is.null(l_contexts)) {
    # remove the contexts
    keep <- c(FALSE, TRUE)
  } else {
    keep <- TRUE
  }
  # split(x, by) |> unsplit(by)
  #   tidytable::map2_dfr(, ~ tidytable::tidytable(x = .x))
  out <- out |> lapply(function(x) x[keep])
   lps <- out |> unsplit(by[keep], drop = TRUE)

   names(lps) <- out |> lapply(function(x) paste0(names(x),"")) |>
     unsplit(by[keep], drop = TRUE)
   lps
  }



#' Get the log probability of each token in a sentence (or group of sentences) using a causal transformer
#'
#' Get the log probability of each token in a sentence (or group of sentences) using a causal transformer model.
#'
#'
#' @param texts Vector or list of texts.
#' @param .id Name of the column with the sentence id.
#' @inheritParams causal_preload
#' @inheritParams causal_lp
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
                                 checkpoint = NULL,
                                 add_special_tokens = NULL,
                                 config_model = NULL,
                                 config_tokenizer = NULL,
                                 batch_size = 1,
                                 .id = NULL) {

 .Deprecated(new = "causal_tokens_pred_tbl()")
  stride <- 1
  message_verbose("Processing using causal model '", file.path(model, checkpoint), "'...")
  ltexts <- as.list(unlist(texts, recursive = TRUE))
  tkzr <- tokenizer(model,
    add_special_tokens = add_special_tokens,
    config_tokenizer = config_tokenizer
  )
  trf <- lang_model(model,
                    checkpoint = checkpoint,
                    task = "causal",
                    config_model = config_model
    )
  tensors <- create_tensor_lst(ltexts,
    tkzr,
    add_special_tokens = add_special_tokens,
    stride = stride,
    batch_size = batch_size
  )

  ls_mat <- tidytable::map(tensors, function(tensor) {
    causal_mat(tensor,
      trf,
      tkzr,
      add_special_tokens = add_special_tokens,
      stride = stride
    )
  }) |>
    unlist(recursive = FALSE)

  tidytable::map_dfr(ls_mat, function(mat) {
    if (ncol(mat) == 1 && colnames(mat) == "") {
      tidytable::tidytable(
        token = "",
        lp = NA_real_
      )
    } else {
      tidytable::tidytable(
        token = colnames(mat),
        lp = tidytable::map2_dbl(colnames(mat), seq_len(ncol(mat)), ~ mat[.x, .y])
      )
    }
  }, .id = .id)
}




#' Get a list of matrices with the log probabilities of possible word given its previous context using a causal transformer
#'
#' Get a list of matrices with the log probabilities of possible word given
#' its previous context using a causal transformer model.
#'
#' @inheritParams causal_lp
#' @inheritParams causal_preload
#' @param sorted When default FALSE it will retain the order of groups we are splitting on. When TRUE then sorted (according to `by`) list(s) are returned.
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
                           by = rep(1, length(x)),
                           sorted = FALSE,
                           model = getOption("pangoling.causal.default"),
                           checkpoint = NULL,
                           add_special_tokens = NULL,
                           config_model = NULL,
                           config_tokenizer = NULL,
                           batch_size = 1,
                           ...) {
 .Deprecated(new = "causal_pred_mats()")
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
  message_verbose("Processing using causal model '", file.path(model, checkpoint), "'...")
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
  pasted_texts <- lapply(
    word_by_word_texts,
    function(word) paste0(word, collapse = " ")
  )
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
    unlist(recursive = FALSE)
}
