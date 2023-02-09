test_that("perplexity works", {
  probs <- c(.3, .5, .6)
  lprobs <- log(probs)
  expect_equal(perplexity(probs, log.p = FALSE),
               perplexity(lprobs, log.p = TRUE))
})
