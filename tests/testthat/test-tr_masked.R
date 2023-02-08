options(pangoling.verbose = FALSE)


# d <- get_masked_tokens_tbl("The apple [MASK][MASK][MASK] fall far from the tree.", model = "bert-large-uncased-whole-word-masking")
#
get_masked_tokens_tbl("[CLS] The apple doesn't fall far from the  [MASK]", model = "bert-large-uncased-whole-word-masking",add_special_tokens = FALSE)
#
# d2 <- get_masked_tokens_tbl("The apple [MASK][MASK][MASK] fall far from the tree.")
#


# sentences <- c("The apple doesn't fall far from the tree.",
#                "Don't judge a book by its cover.")
# df_sent <- strsplit(x = sentences, split = " ") |>
#   tidytable::map_dfr(.f =  ~ data.frame(word = .x), .id = "sent_n")
# df_sent
#
# df_sent <- df_sent |>
#   tidytable::mutate(lp = get_masked_log_prob(word, model = "bert-large-uncased-whole-word-masking", .by = sent_n))
# df_sent

test_that("bert masked works", {
  skip_if_no_python_stuff()
  mask_1 <-
    get_masked_tokens_tbl("The apple doesn't fall far from the [MASK].", model = "google/bert_uncased_L-2_H-128_A-2")

  expect_snapshot(mask_1)
  mask_2 <-
    get_masked_tokens_tbl("The apple doesn't fall far from [MASK] [MASK].", model = "google/bert_uncased_L-2_H-128_A-2")
  expect_snapshot(mask_2)
  mask_2_ <-
    get_masked_tokens_tbl("[CLS] The apple doesn't fall far from [MASK] [MASK]. [SEP]", model = "google/bert_uncased_L-2_H-128_A-2", add_special_tokens = FALSE)
  expect_equal(mask_2, mask_2_)

  nomask <- get_masked_tokens_tbl("Don't judge a book by its cover.", model = "google/bert_uncased_L-2_H-128_A-2")
  masks_2_nomask <- get_masked_tokens_tbl(masked_sentences = c("The apple doesn't fall far from [MASK] [MASK].", "Don't judge a book by its [MASK].","Don't judge a book by its cover."), model = "google/bert_uncased_L-2_H-128_A-2")

  expect_equal(mask_2, masks_2_nomask |>
                 tidytable::filter(masked_sentence == "The apple doesn't fall far from [MASK] [MASK]."))

})

