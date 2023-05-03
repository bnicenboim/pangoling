#' Returns the vocabulary of a model
#'
#' Returns the vocabulary of a model.
#'
#' @inheritParams causal_lp
#'
#' @return A vector with the vocabulary of a model.
#' @examplesIf interactive()
#' transformer_vocab(model = "gpt2")
#' @export
#'
#' @family token-related functions
transformer_vocab <- function(model = getOption("pangoling.causal.default"),
                              add_special_tokens = NULL,
                              config_tokenizer = NULL) {
  tkzr <- tokenizer(model,
    add_special_tokens = add_special_tokens,
    config_tokenizer = config_tokenizer
  )
  sort(unlist(tkzr$get_vocab())) |> names()
}

#' Tokenize an input
#'
#' Tokenize a string or token ids.
#'
#' @param x Strings or token ids.
#' @inheritParams causal_lp
#' @return A list with tokens
#'
#' @examplesIf interactive()
#' tokenize_lst(x = c("The apple doesn't fall far from the tree."), model = "gpt2")
#' @family token-related functions
#' @export
tokenize_lst <- function(x,
                         model = getOption("pangoling.causal.default"),
                         add_special_tokens = NULL,
                         config_tokenizer = NULL) {
  UseMethod("tokenize_lst")
}

#' @export
tokenize_lst.character <- function(x,
                                   model = getOption("pangoling.causal.default"),
                                   add_special_tokens = NULL,
                                   config_tokenizer = NULL) {
  id <- get_id(x,
    model = model,
    add_special_tokens = add_special_tokens,
    config_tokenizer = config_tokenizer
  )
  lapply(id, function(i) {
    tokenize_lst.numeric(i,
      model = model,
      add_special_tokens = add_special_tokens,
      config_tokenizer = config_tokenizer
    )
  })
}

#' @export
tokenize_lst.numeric <- function(x,
                                 model = getOption("pangoling.causal.default"),
                                 add_special_tokens = NULL,
                                 config_tokenizer = NULL) {
  tidytable::map_chr(as.integer(x), function(x) {
    tokenizer(model,
      add_special_tokens = add_special_tokens,
      config_tokenizer = config_tokenizer
    )$convert_ids_to_tokens(x)
  })
}

#' The number of tokens in a string or vector of strings
#'
#' @param x character input
#' @inheritParams tokenize_lst
#'
#' @return The number of tokens in a string or vector of words.
#'
#'
#' @examplesIf interactive()
#' ntokens(x = c("The apple doesn't fall far from the tree."), model = "gpt2")
#' @family token-related functions
#' @export
ntokens <- function(x,
                    model = getOption("pangoling.causal.default"),
                    add_special_tokens = NULL,
                    config_tokenizer = NULL) {
  lengths(tokenize_lst(x,
    model = model,
    add_special_tokens = add_special_tokens,
    config_tokenizer = config_tokenizer
  ))
}


get_vocab <- function(tkzr) {
  sort(unlist(tkzr$get_vocab())) |> names()
}

encode <- function(x, tkzr, add_special_tokens = NULL, ...) {
  if (!is.null(add_special_tokens)) {
    tkzr$batch_encode_plus(x,
      return_tensors = "pt",
      add_special_tokens = add_special_tokens, ...
    )
  } else {
    tkzr$batch_encode_plus(x, return_tensors = "pt", ...)
  }
}

get_word_by_word_texts <- function(x, .by) {
  if (length(x) != length(.by)) {
    stop2("The argument `.by` has an incorrect length.")
  }
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
lang_model <- function(model = "gpt2", checkpoint = NULL, task = "causal", config_model = NULL) {
  reticulate::py_run_string(
    'import os\nos.environ["TOKENIZERS_PARALLELISM"] = "false"'
  )
  if(length(checkpoint)>0 && checkpoint != ""){
    model <- file.path(model, checkpoint)
  }
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
  lst_to_kwargs(c(
    pretrained_model_name_or_path = model,
    return_dict_in_generate = TRUE,
    config_model
  ))
  reticulate::py_run_string(paste0(
    "lm = transformers.",
    automodel,
    ".from_pretrained(**r.kwargs)"
  ))

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
tokenizer <- function(model = "gpt2",
                      add_special_tokens = NULL,
                      config_tokenizer = NULL) {
  reticulate::py_run_string("import transformers")
  if (chr_detect(model, "gpt2") && !is.null(add_special_tokens)) {
    lst_to_kwargs(c(
      pretrained_model_name_or_path = model,
      add_bos_token = add_special_tokens,
      config_tokenizer
    ))
    reticulate::py_to_r(
      reticulate::py_run_string(
        "tkzr = transformers.GPT2Tokenizer.from_pretrained(**r.kwargs)"
      )
    )
  } else {
    lst_to_kwargs(c(pretrained_model_name_or_path = model, config_tokenizer))
    reticulate::py_to_r(
      reticulate::py_run_string(
        "tkzr = transformers.AutoTokenizer.from_pretrained(**r.kwargs)"
      )
    )
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
get_id <- function(x,
                   model = "gpt2",
                   add_special_tokens = NULL,
                   config_tokenizer = NULL,
                   tkzr = NULL) {
  if (is.null(tkzr)) {
    tkzr <- tokenizer(model,
      add_special_tokens = add_special_tokens,
      config_tokenizer = config_tokenizer
    )
  }
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
create_tensor_lst <- function(texts,
                              tkzr,
                              add_special_tokens = NULL,
                              stride = 1,
                              batch_size = 1) {
  if (is.null(tkzr$special_tokens_map$pad_token) &&
    !is.null(tkzr$special_tokens_map$eos_token)) {
    tkzr$pad_token <- tkzr$eos_token
  }
  texts <- unlist(texts)
  # If I runt the following line, some models such as
  # 'flax-community/gpt-2-spanish' give a weird error of
  # 'GPT2TokenizerFast' object has no attribute 'is_fast'
  # max_length <- tkzr$model_max_length
  # thus the ugly hack
  # max_length <- chr_match(utils::capture.output(tkzr),
  #                         pattern = "model_max_len=([0-9]*)") |>
  #   c() |>
  # (\(x) x[[2]])()
  # if (is.null(max_length) || is.na(max_length) || max_length < 1) {
  #   message_verbose("Unknown maximum length of input.
  # This might cause a problem for long inputs exceeding the maximum length.")
  #   max_length <- Inf
  # }

  g_batches <- c(rep(batch_size, floor(length(texts) / batch_size)), length(texts) %% batch_size)
  g_batches <- g_batches[g_batches > 0]
  text_ids <- tidytable::map2(
    c(1, cumsum(g_batches)[-length(g_batches)] + 1), cumsum(g_batches),
    ~ seq(.x, .y)
  )
  lapply(text_ids, function(text_id) {
    # message(paste(text_id, " "))
    tensor <- encode(
      x = as.list(texts[text_id]),
      tkzr = tkzr,
      add_special_tokens = add_special_tokens,
      stride = as.integer(stride),
      truncation = TRUE, # is.finite(max_length),
      return_overflowing_tokens = FALSE, # is.finite(max_length),
      padding = TRUE # is.finite(max_length)
    )
    tensor
  })
}


word_lp <- function(words,
                    mat,
                    ignore_regex,
                    model,
                    add_special_tokens,
                    config_tokenizer) {
  if (length(words) == 1 && words == "") {
    return(NA_real_)
  }
  if (length(words) > 1) {
    words_lm <- c(words[1], paste0(" ", words[-1]))
  } else {
    words_lm <- words
  }
  tokens <- lapply(get_id(words_lm,
    model,
    add_special_tokens = add_special_tokens,
    config_tokenizer = config_tokenizer
  ),
  tokenize_lst.numeric,
  model = model,
  add_special_tokens = add_special_tokens,
  config_tokenizer = config_tokenizer
  )
  token_n <- tidytable::map_dbl(tokens, length)
  index_vocab <- data.table::chmatch(unlist(tokens), rownames(mat))


  token_lp <- tidytable::map2_dbl.(index_vocab, seq_len(ncol(mat)), ~ mat[.x, .y])

  if (options()$pangoling.debug) {
    print("******")
    sent <- tidytable::map_chr(tokens, function(x) paste0(x, collapse = "|"))
    print(paste0("[", sent, "]", collapse = " "))
    print(token_lp)
  }
  if (length(ignore_regex) > 0 && ignore_regex != "") {
    pos <- which(grepl(pattern = ignore_regex, x = unlist(tokens)))
    token_lp[pos] <- 0
  }
  # ignores the NA in the first column if it starts with a special character
  if (unlist(tokens)[1] %in% tokenizer(model)$all_special_tokens) {
    token_lp[1] <- 0
  }
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

char_to_token <- function(x, tkzr = NULL) {
  tokenizer <- tkzr
  id <- get_id(x, tkzr = tokenizer)
  lapply(id, function(i) num_to_token(i, tokenizer))
}

num_to_token <- function(x, tkzr) {
  tidytable::map_chr(as.integer(x), function(x) {
    tkzr$convert_ids_to_tokens(x)
  })
}
