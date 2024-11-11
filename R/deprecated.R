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
 .Deprecated("sum")

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
  word_by_word_texts <- get_word_by_word_texts(x, by)

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
