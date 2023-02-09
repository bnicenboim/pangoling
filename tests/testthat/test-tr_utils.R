options(pangoling.verbose = FALSE)

test_that("multiplication works", {
  expect_true(is.character(transformer_vocab()))
  sents <- c("This is a sentence.", "This ain't a sentence.")
  expect_equal(lengths(tokenize(sents)), ntokens(sents))
})
