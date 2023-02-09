options(pangoling.verbose = FALSE)



test_that("masked load and gets config", {
  expect_invisible(masked_preload())
  conf_lst <- masked_config()
  expect_true(is.list(conf_lst))
  expect_equal(conf_lst$`_name_or_path`, getOption("pangoling.masked.default"))
})


test_that("bert masked works", {
  skip_if_no_python_stuff()
  mask_1 <-
    masked_tokens_tbl("The apple doesn't fall far from the [MASK].", model = "google/bert_uncased_L-2_H-128_A-2")

  expect_snapshot(mask_1)
  mask_2 <-
    masked_tokens_tbl("The apple doesn't fall far from [MASK] [MASK].", model = "google/bert_uncased_L-2_H-128_A-2")
  expect_snapshot(mask_2)
  mask_2_ <-
    masked_tokens_tbl("[CLS] The apple doesn't fall far from [MASK] [MASK]. [SEP]", model = "google/bert_uncased_L-2_H-128_A-2", add_special_tokens = FALSE)
  expect_equal(mask_2[,-1], mask_2_[,-1])

  nomask <- masked_tokens_tbl("Don't judge a book by its cover.", model = "google/bert_uncased_L-2_H-128_A-2")
  masks_2_nomask <- masked_tokens_tbl(masked_sentences = c("The apple doesn't fall far from [MASK] [MASK].", "Don't judge a book by its [MASK].","Don't judge a book by its cover."), model = "google/bert_uncased_L-2_H-128_A-2")

  expect_equal(mask_2, masks_2_nomask |>
                 tidytable::filter(masked_sentence == "The apple doesn't fall far from [MASK] [MASK]."))

})

test_that("bert last word works", {
   lw <- masked_last_lp(c("The apple doesn't fall far from the",
                     "The tree doesn't fall far from the"),
                last_words = c("tree","apple"),
                  model = "google/bert_uncased_L-2_H-128_A-2")

})
model = "google/bert_uncased_L-2_H-128_A-2"
tkzr <- tokenizer(model,
                  add_special_tokens = NULL,
                  config_tokenizer = NULL)
tkzr$encode("it", return_tensors = "pt")
encode("it",tkzr)
# AttributeError: 'BertTokenizerFast' object has no attribute 'encode'
masked_sentences <- "bb"
last_words <- "Asd"
  l <- create_tensor_lst("pp",
                         tkzr,
                         add_special_tokens = NULL,
                         stride = 1)

  if (is.null(tkzr$special_tokens_map$pad_token) &&
      !is.null(tkzr$special_tokens_map$eos_token)) {
    tkzr$pad_token <- tkzr$eos_token
  }
  max_length <- tkzr$max_len_single_sentence
  if (is.null(max_length) || is.na(max_length) || max_length < 1) {
    warning("Unknown maximum length of input. This might cause a problem for long inputs exceeding the maximum length.")
    max_length <- Inf
  }

xx <- function(texts,
               tkzr,
               add_special_tokens = NULL,
               stride = 1,
               max_length = 512) {
  if (is.null(tkzr$special_tokens_map$pad_token) &&
      !is.null(tkzr$special_tokens_map$eos_token)) {
    tkzr$pad_token <- tkzr$eos_token
  }

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
xx(texts = "a",tkzr,add_special_tokens = NULL,stride = 1)
