test_that("perplexity_calc works", {
  probs <- c(.3, .5, .6)
  lprobs <- log(probs)
  l2probs <- log(probs, base = 2)
  expect_equal(
    perplexity_calc(probs, log.p = FALSE),
    perplexity_calc(lprobs, log.p = TRUE)
  )
  expect_equal(
    perplexity_calc(probs, log.p = FALSE),
    perplexity_calc(l2probs, log.p = 2)
  )
})
