options(pangoling.verbose = FALSE)

test_that("multiplication works", {
  skip_if_no_python_stuff()
  expect_true(is.character(transformer_vocab()))
  sents <- c("This is a sentence.", "This ain't a sentence.")
  expect_equal(lengths(tokenize(sents)), ntokens(sents))
})
