options(pangoling.verbose = FALSE)

test_that("utils works", {
  skip_if_no_python_stuff()
  expect_true(is.character(transformer_vocab()))
  sents <- c("This is a sentence.", "This ain't a sentence.")
  expect_equal(lengths(tokenize_lst(sents)), ntokens(sents))
  expect_equal(tokenize_lst(sents, decode = TRUE), 
               list(c("This", " is", " a", " sentence", "."), 
                    c("This", " ain", "'t", " a", " sentence", ".")))
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


test_that("pangoling:::is_really_string handles various cases correctly", {
  expect_true(pangoling:::is_really_string("hello"))
  expect_equal(pangoling:::is_really_string(c("a", "b", "c")), c(TRUE, TRUE, TRUE))
  expect_equal(pangoling:::is_really_string(c("", "a", "b")), c(FALSE, TRUE, TRUE))
  expect_equal(pangoling:::is_really_string(c(NA, "a", "b")), c(FALSE, TRUE, TRUE))
  expect_false(pangoling:::is_really_string(NULL))
  expect_equal(pangoling:::is_really_string(c(1, 2, 3)), c(FALSE, FALSE, FALSE))
  expect_equal(pangoling:::is_really_string(c(TRUE, FALSE)), c(FALSE, FALSE))
  expect_false(pangoling:::is_really_string(character(0)))
  expect_equal(pangoling:::is_really_string(c("", "a", NA, "b", NULL)), c(FALSE, TRUE, FALSE, TRUE))
})
