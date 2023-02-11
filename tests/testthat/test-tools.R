test_that("perplexity works", {
  probs <- c(.3, .5, .6)
  lprobs <- log(probs)
  l2probs <- log(probs, base = 2)
  expect_equal(
    perplexity(probs, log.p = FALSE),
    perplexity(lprobs, log.p = TRUE)
  )
  expect_equal(
    perplexity(probs, log.p = FALSE),
    perplexity(l2probs, log.p = 2)
  )
})
