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
  expect_equal(colnames(mask_1), c("masked_sentence", "token", "pred", "mask_n"))
  expect_equal(sum(exp(mask_1$pred)), 1, tolerance = 0.0001)
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
    prev_contexts = c("The", "The"),
    targets = c("apple", "pear"),
    after_contexts = c(
      "doesn't fall far from the tree.",
      "doesn't fall far from the tree."
    ),
    model = "google/bert_uncased_L-2_H-128_A-2"
  )
  ms <- masked_tokens_pred_tbl(c("The [MASK] doesn't fall far from the tree."),
    model = "google/bert_uncased_L-2_H-128_A-2"
  )
  lps <- c(
    ms[token == "apple", ]$pred,
    ms[token == "pear", ]$pred
  )
  names(lps) <- c("apple", "pear")
  expect_equal(lw, lps)
# two tokens target
  lw <- masked_targets_pred(
    prev_contexts = c("The"),
    targets = c("tasty"),
    after_contexts = c(
      "lunch."
    ),
    model = "google/bert_uncased_L-2_H-128_A-2"
  )
ms <- masked_tokens_pred_tbl(c("The [MASK] [MASK] lunch."),
    model = "google/bert_uncased_L-2_H-128_A-2"
  )
  lps <- c(
    ms[token == "ta" & mask_n ==1, ]$pred+   ms[token == "##sty" & mask_n ==2, ]$pred
  )
  names(lps) <- c("tasty")
  expect_equal(lw, lps)


})


test_that("bert lp for multiple target words works", {
  skip_if_no_python_stuff()
  lw <- masked_targets_pred(
    prev_contexts = c("The", "The"),
    targets = c("nice apple", "pretty pear"),
    after_contexts = c(
      "doesn't fall far from the tree.",
      "doesn't fall far from the tree."
    ),
    model = "google/bert_uncased_L-2_H-128_A-2"
  )
  ms <- masked_tokens_pred_tbl(c("The [MASK] [MASK] doesn't fall far from the tree."),
    model = "google/bert_uncased_L-2_H-128_A-2"
  )
  lps <- c(
    ms[token == "nice" & mask_n==1, ]$pred + ms[token == "apple" & mask_n==2, ]$pred,
    ms[token == "pretty" & mask_n==1, ]$pred + ms[token == "pear" & mask_n==2, ]$pred
  )
  names(lps) <- c("nice apple", "pretty pear")
  expect_equal(lw, lps)

lw <- masked_targets_pred(
    prev_contexts = c("The", "The"),
    targets = c("nice apple", "tasty pear"),
    after_contexts = c(
      "doesn't fall far from the tree.",
      "doesn't fall far from the tree."
    ),
    model = "google/bert_uncased_L-2_H-128_A-2"
  )

   ms2 <- masked_tokens_pred_tbl(c("The [MASK] [MASK] [MASK] doesn't fall far from the tree."),
    model = "google/bert_uncased_L-2_H-128_A-2"
  )
  lps2 <- c(
    ms2[token == "ta" & mask_n==1, ]$pred + ms2[token == "##sty" & mask_n ==2, ]$pred + ms2[token == "pear" & mask_n==3, ]$pred
  )
names(lps2) <- "tasty pear"

  expect_equal(lw, c(lps[1], lps2))
})


test_that("bert works in hebrew", {
  expect_no_error(masked_tokens_pred_tbl(
  masked_sentences =  "אני אוהב  [MASK].",
  model = "onlplab/alephbert-base"
))


expect_no_error(lw <- masked_targets_pred(
  prev_contexts = c("אני אוהב", "אני אוהב"),
  targets = c("אותך", "אותה"),
  after_contexts = c(
    ".",
    "."
  ),
  model = "onlplab/alephbert-base"
))

})
