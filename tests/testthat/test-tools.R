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


test_that("ln_p_change handles TRUE, FALSE, and numeric log.p correctly", {
  # Input log probabilities
  x <- c(-2.302585, -1.609438, -0.693147, 0) # log probabilities
    result <- pangoling:::ln_p_change(x, log.p = TRUE)
  expect_equal(result, x)  
  result <- pangoling:::ln_p_change(x, log.p = FALSE)
  expect_equal(result, exp(x))
  result <- pangoling:::ln_p_change(x, log.p = 2)
  expect_equal(result, x / log(2))
})

test_that("ln_p_change handles edge cases", {
  result <- pangoling:::ln_p_change(numeric(0), log.p = TRUE)
  expect_equal(result, numeric(0))  
  x <- c(-2.302585, NA, -0.693147)
  result <- pangoling:::ln_p_change(x, log.p = TRUE)
  expect_equal(result, x)  
})
