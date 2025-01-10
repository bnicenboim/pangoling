options(pangoling.verbose = FALSE)

test_that("utils works", {
  skip_if_no_python_stuff()
  expect_true(is.character(transformer_vocab()))
  sents <- c("This is a sentence.", "This ain't a sentence.")
  expect_equal(lengths(tokenize_lst(sents)), ntokens(sents))
  expect_equal(tokenize_lst(sents, decode = TRUE), list(c("This", " is", " a", " sentence", "."), c("This", " ain", 
                                                                                                    "'t", " a", " sentence", ".")))
  })

test_that("messages work", {
  skip_if_no_python_stuff()
  options(pangoling.verbose = TRUE)
  expect_message(causal_preload(),
    regexp = paste0(
      ".*?",
      getOption("pangoling.causal.default"), "..."
    )
  )
  options(pangoling.verbose = FALSE)
  expect_no_message(causal_preload())
})

message("TEST cache")
