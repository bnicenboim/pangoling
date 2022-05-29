test_that("multiplication works", {
  expect_equal(get_word_fpmw(x = c("banana","cat","car")),
  get_word_feature(x = c("banana","cat","car"), feature = "fpmw"))
  expect_equal(LexOPS::lexops %>% tidytable::filter.(string=="car") %>% .$Zipf.SUBTLEX_US,
               unname(get_word_feature(x = c("car"))))
})
