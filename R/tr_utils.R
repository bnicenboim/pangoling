#' Returns the vocabulary of a model
#'
#' @inheritParams causal_lp
#'
#' @return A vector with the vocabulary of a model
#' @export
#'
#'
transformer_vocab <- function(model = "gpt2",
                              add_special_tokens = NULL,
                              config_tokenizer = NULL) {
  tkzr <- tokenizer(model,
                    add_special_tokens = add_special_tokens,
                    config_tokenizer = config_tokenizer)
}

#' Tokenize the input
#'
#' @param x Strings or token ids.
#' @inheritParams causal_lp
#' @param config List with arguments that control how the tokenizer from hugging face is accessed.
#' @return
#' @export
#'
#' @examples
tokenize <- function(x, model = "gpt2", add_special_tokens = NULL, config_tokenizer = NULL) {
  UseMethod("tokenize")
}

#' @export
tokenize.character <- function(x, model = "gpt2", add_special_tokens = NULL, config_tokenizer = NULL) {
  id <- get_id(x, model = model, add_special_tokens = add_special_tokens, config_tokenizer = config_tokenizer)
  lapply(id, function(i) tokenize.numeric(i, model = model))
}

#' @export
tokenize.numeric <- function(x, model = "gpt2", add_special_tokens = NULL, config_tokenizer = NULL) {
  tidytable::map_chr.(as.integer(x), function(x) {
    tokenizer(model, add_special_tokens = add_special_tokens, config_tokenizer = config_tokenizer)$convert_ids_to_tokens(x)
  })
}

#' The number of tokens in a string or vector of strings
#'
#' @param x character input
#' @inheritParams tokenize
#'
#' @return The number of tokens in a string or vector of words.
#'
#' @export
#'
#' @examples
ntokens <- function(x, model = "gpt2", add_special_tokens = NULL, config_tokenizer = NULL) {
  length(unlist(tokenize(x, model, add_special_tokens = add_special_tokens, config_tokenizer = config_tokenizer), recursive = TRUE))
}


get_vocab <- function(tkzr) {
  vocab <- sort(unlist(tkzr$get_vocab())) |> names()
}

encode <- function(x, tkzr, add_special_tokens = NULL, ...) {
  if (!is.null(add_special_tokens)) {
    tkzr$encode(x,
      return_tensors = "pt",
      add_special_tokens = add_special_tokens, ...
    )
  } else {
    tkzr$encode(x, return_tensors = "pt", ...)
  }
}

get_word_by_word_texts <- function(x, .by) {
  if (length(x) != length(.by)) stop2("The argument `.by` has an incorrect length.")
  x <- trimws(x, whitespace = "[ \t]")
  split(x, .by)
}

#' Sends a var to python
#' https://stackoverflow.com/questions/67562889/interoperability-between-python-and-r
#' @noRd
var_to_py <- function(var_name, x) {
  e <- new.env()
  options("reticulate.engine.environment" = e)
  assign(var_name, x, envir = e)
  # options("reticulate.engine.environment" = NULL)
}

lst_to_kwargs <- function(x) {
  x <- x[lengths(x) > 0]
  if (!is.list(x)) x <- as.list(x)
  x <- reticulate::r_to_py(x)
  var_to_py("kwargs", x)
}

#' @noRd
lang_model <- function(model = "gpt2", task = "causal", config_model = NULL) {
  reticulate::py_run_string('import os\nos.environ["TOKENIZERS_PARALLELISM"] = "false"')

  # to prevent memory leaks:
  reticulate::py_run_string('there = "lm" in locals()')
  if (reticulate::py$there) reticulate::py_run_string("del lm")
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
    causal = "AutoModelForCausalLM",
    masked = "AutoModelForMaskedLM"
  )
  lst_to_kwargs(c(pretrained_model_name_or_path = model, return_dict_in_generate = TRUE, config_model))
  reticulate::py_run_string(paste0("lm = transformers.", automodel, ".from_pretrained(**r.kwargs)"))

  lm <- reticulate::py$lm
  lm$eval()

  options("reticulate.engine.environment" = NULL) # unset option
  # trys to remove everything from memory
  reticulate::py_run_string("import gc
gc.collect()")
  gc(full = TRUE)

  lm
}

#' https://huggingface.co/docs/transformers/v4.25.1/en/model_doc/auto#transformers.AutoTokenizer
#' @noRd
tokenizer <- function(model = "gpt2", add_special_tokens = NULL, config_tokenizer = NULL) {
  reticulate::py_run_string("import transformers")
  if (chr_detect(model, "gpt2") && !is.null(add_special_tokens)) {
    lst_to_kwargs(c(pretrained_model_name_or_path = model, add_bos_token = add_special_tokens, config_tokenizer))
    reticulate::py_to_r(reticulate::py_run_string("tkzr = transformers.GPT2Tokenizer.from_pretrained(**r.kwargs)"))
    #
    # tkzr <- transformers$GPT2Tokenizer$from_pretrained(model,add_bos_token = add_bos_token, ...)
  } else {
    lst_to_kwargs(c(pretrained_model_name_or_path = model, config_tokenizer))

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







#' Get ids (without adding special characters at beginning or end?)
#' @noRd
get_id <- function(x, model = "gpt2", add_special_tokens = add_special_tokens, config_tokenizer = NULL) {
  tkzr <- tokenizer(model, add_special_tokens = add_special_tokens, config_tokenizer = config_tokenizer)
  if (!is.null(add_special_tokens) && add_special_tokens) {
    x[1] <- paste0(
      tkzr$special_tokens_map$bos_token,
      tkzr$special_tokens_map$cls_token, x[1]
    )
    x[length(x)] <- paste0(x[length(x)], tkzr$special_tokens_map$sep_token)
  } ### more general
  lapply(x, function(i) {
    t <- tkzr$tokenize(i)
    tkzr$convert_tokens_to_ids(t)
  })
}




#' @noRd
get_lm_lp <- function(x, by = rep(1, length(x)), ignore_regex = "", type = "causal", model = "gpt2", n_plus = 3, ...) {
  if (length(x) != length(by)) stop2("The argument `by` has an incorrect length.")
  if (length(x) <= 1) stop2("The argument `x` needs at least two elements.")
  x <- trimws(x, whitespace = "[ \t]")
  texts <- split(x, by)
  N <- length(texts)
  out <- tidytable::map2.(texts, names(texts), function(words, item) {
    # words <- texts[[1]]
    # item <- names(texts[1])
    if (type == "causal") ls_mat <- causal_lp_mat(words, model = model, ...)
    if (type == "masked") ls_mat <- masked_lp_mat(words, model = model, n_plus = n_plus, ...)


    if (length(words) > 1) {
      words_lm <- c(words[1], paste0(" ", words[-1]))
    } else {
      words_lm <- words
    }

    tokens <- lapply(get_id(words_lm, model), tokenize.numeric, model = model)
    token_n <- tidytable::map_dbl.(tokens, length)
    index_vocab <- data.table::chmatch(unlist(tokens), vocab)
    message_verbose("Text id: ", item, "\n`", paste(words, collapse = " "), "`")
    # if(length(ls_mat)>1){
    # # when there is a matrix for the predictions made from each word,
    # # remove the predictions made in a token in the middle of a word:
    # # for example from 'nt
    # token_remove <- tidytable::map.(tokens, ~ cumsum(seq_along(.x)) >1) |>
    #   unlist()
    # ls_mat[token_remove] <- NULL
    # }

    # predictions for word n+1, n+2, etc...
    if (n_plus == 0) n_plus <- length(ls_mat)
    ls_preds <- lapply(1:n_plus, function(p) {
      # when p = 1, predictions for token[i] are made in token[i-1]
      mat <- ls_mat[[p]]
      # mat[vocab =="isn",2]
      token_lp <- tidytable::map2_dbl.(index_vocab, 1:ncol(mat), ~ mat[.x, .y])

      if (options()$pangoling.debug) {
        print("******")
        sent <- tidytable::map_chr.(tokens, function(x) paste0(x, collapse = "|"))
        print(paste0("[", sent, "]", collapse = " "))
        print(token_lp)
      }
      n <- 1
      word_lp <- vector(mode = "numeric", length(words))
      # It might ignore punctuation if ignore_regex is used
      if (length(ignore_regex) > 0 && ignore_regex != "") {
        pos <- which(grepl(pattern = ignore_regex, x = unlist(tokens)))
        token_lp[pos] <- 0
      }
      # ignores the NA in the first column if it starts with a special character
      if (unlist(tokens)[1] %in% reticulate::py_to_r(tokenizer(model)$all_special_tokens)) token_lp[1] <- 0
      # i <- 1
      for (i in seq_along(token_n)) {
        t <- token_n[i]
        n_p <- n - (p - 1) # source of the prediction
        if (n_p < 1 || !n_p %in% c(cumsum(c(0, token_n)) + 1)) {
          word_lp[i] <- NA
        } else {
          word_lp[i] <- sum(token_lp[n:(n + (t - 1))])
        }
        n <- n + t
        # i <- i + 1
      }
      word_lp
    })
    pred_mat <- do.call("rbind", ls_preds)
    # change order so that they start from pred n+1
    # pred_mat <- pred_mat[nrow(pred_mat):1, , drop = FALSE]
    colnames(pred_mat) <- words_lm
    lapply(as.list(as.data.frame(pred_mat)), function(x) {
      # removes the NA from the front
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
slide_tokens <- function(input_ids, max_tokens, stride) {
  message_verbose("Number of tokens larger than the maximum allowed ", max_tokens, ". Using a sliding window.")
  # build a matrix with max tokens rows, and as many columns as needed
  ids_matrix <- embed(input_ids, max_tokens)[, max_tokens:1]
  rel_rows <- c(seq(1, nrow(ids_matrix) - 1, stride), nrow(ids_matrix))
  rel_token_pos <- diff(rel_rows)
  ids_matrix <- ids_matrix[rel_rows, ]
  lapply(seq_len(nrow(ids_matrix)), function(i) ids_matrix[i, ])
}

#' @noRd
rel_pos_slide <- function(input_ids, max_tokens, stride) {
  ids_matrix <- embed(input_ids, max_tokens)[, max_tokens:1]
  rel_rows <- c(seq(1, nrow(ids_matrix) - 1, stride), nrow(ids_matrix))
  c(list(1:max_tokens), lapply(diff(rel_rows), function(.x) seq.int(from = max_tokens - .x + 1, to = max_tokens)))
}

#' @noRd
create_tensor_lst <- function(texts,
                              tkzr,
                              add_special_tokens = NULL,
                              stride = 1) {
  if (is.null(tkzr$special_tokens_map$pad_token) &&
    !is.null(tkzr$special_tokens_map$eos_token)) {
    tkzr$pad_token <- tkzr$eos_token
  }
  max_length <- tkzr$max_len_single_sentence
  if (is.null(max_length) || is.na(max_length) || max_length < 1) {
    warning("Unknown maximum length of input. This might cause a problem for long inputs exceeding the maximum length.")
    max_length <- Inf
  }
  lapply(texts, function(text) {
    tensor <- encode(text,
      tkzr,
      add_special_tokens = add_special_tokens,
      stride = as.integer(stride),
      truncation = is.finite(max_length),
      return_overflowing_tokens = is.finite(max_length),
      padding = is.finite(max_length)
    )
    tensor
  })
}



word_lp <- function(words, mat, ignore_regex, model, add_special_tokens, config_tokenizer) {
  if (length(words) > 1) {
    words_lm <- c(words[1], paste0(" ", words[-1]))
  } else {
    words_lm <- words
  }
  tokens <- lapply(get_id(words_lm, model, add_special_tokens = add_special_tokens, config_tokenizer = config_tokenizer),
    tokenize.numeric,
    model = model, add_special_tokens = add_special_tokens, config_tokenizer = config_tokenizer
  )
  token_n <- tidytable::map_dbl.(tokens, length)
  index_vocab <- data.table::chmatch(unlist(tokens), rownames(mat))


  token_lp <- tidytable::map2_dbl.(index_vocab, 1:ncol(mat), ~ mat[.x, .y])

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
}
