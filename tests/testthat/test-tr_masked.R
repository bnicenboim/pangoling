options(pangoling.verbose = FALSE)



test_that("masked load and gets config", {
  skip_if_no_python_stuff()
  expect_invisible(masked_preload())
  conf_lst <- masked_config()
  expect_true(is.list(conf_lst))
  expect_equal(conf_lst$`_name_or_path`, getOption("pangoling.masked.default"))
})


test_that("bert masked works", {
  skip_if_no_python_stuff()
  mask_1 <-
    masked_tokens_pred_tbl(
      masked_sentences = "The apple doesn't fall far from the [MASK].",
      model = "google/bert_uncased_L-2_H-128_A-2"
    )
  expect_equal(colnames(mask_1), c("masked_sentence", "token", "lp", "mask_n"))
  expect_equal(sum(exp(mask_1$lp)), 1, tolerance = 0.0001)
  mask_2 <-
    masked_tokens_pred_tbl("The apple doesn't fall far from [MASK] [MASK].",
      model = "google/bert_uncased_L-2_H-128_A-2"
    )
  mask_2_ <-
    masked_tokens_pred_tbl(
      "[CLS] The apple doesn't fall far from [MASK] [MASK]. [SEP]",
      model = "google/bert_uncased_L-2_H-128_A-2",
      add_special_tokens = FALSE
    )
  expect_equal(mask_2[, -1], mask_2_[, -1])

  nomask <- masked_tokens_pred_tbl("Don't judge a book by its cover.",
    model = "google/bert_uncased_L-2_H-128_A-2"
  )
  masks_2_nomask <- masked_tokens_pred_tbl(
    masked_sentences = c(
      "The apple doesn't fall far from [MASK] [MASK].",
      "Don't judge a book by its [MASK].",
      "Don't judge a book by its cover."
    ),
    model = "google/bert_uncased_L-2_H-128_A-2"
  )

  expect_equal(
    mask_2,
    masks_2_nomask |>
      tidytable::filter(masked_sentence ==
        "The apple doesn't fall far from [MASK] [MASK].")
  )
})

test_that("bert lp for target words works", {
  skip_if_no_python_stuff()
  lw <- masked_targets_pred(
    l_contexts = c("The", "The"),
    targets = c("apple", "pear"),
    r_contexts = c(
      "doesn't fall far from the tree.",
      "doesn't fall far from the tree."
    ),
    model = "google/bert_uncased_L-2_H-128_A-2"
  )
  ms <- masked_tokens_pred_tbl(c("The [MASK] doesn't fall far from the tree."),
    model = "google/bert_uncased_L-2_H-128_A-2"
  )
  lps <- c(
    ms[token == "apple", ]$lp,
    ms[token == "pear", ]$lp
  )
  names(lps) <- c("apple", "pear")
  expect_equal(lw, lps)
# two tokens target
  lw <- masked_targets_pred(
    l_contexts = c("The"),
    targets = c("tasty"),
    r_contexts = c(
      "lunch."
    ),
    model = "google/bert_uncased_L-2_H-128_A-2"
  )
ms <- masked_tokens_pred_tbl(c("The [MASK] [MASK] lunch."),
    model = "google/bert_uncased_L-2_H-128_A-2"
  )
  lps <- c(
    ms[token == "ta" & mask_n ==1, ]$lp+   ms[token == "##sty" & mask_n ==2, ]$lp
  )
  names(lps) <- c("tasty")
  expect_equal(lw, lps)


})


test_that("bert lp for multiple target words works", {
  skip_if_no_python_stuff()
  lw <- masked_targets_pred(
    l_contexts = c("The", "The"),
    targets = c("nice apple", "pretty pear"),
    r_contexts = c(
      "doesn't fall far from the tree.",
      "doesn't fall far from the tree."
    ),
    model = "google/bert_uncased_L-2_H-128_A-2"
  )
  ms <- masked_tokens_pred_tbl(c("The [MASK] [MASK] doesn't fall far from the tree."),
    model = "google/bert_uncased_L-2_H-128_A-2"
  )
  lps <- c(
    ms[token == "nice" & mask_n==1, ]$lp + ms[token == "apple" & mask_n==2, ]$lp,
    ms[token == "pretty" & mask_n==1, ]$lp + ms[token == "pear" & mask_n==2, ]$lp
  )
  names(lps) <- c("nice apple", "tasty pear")
  expect_equal(lw, lps)
})
