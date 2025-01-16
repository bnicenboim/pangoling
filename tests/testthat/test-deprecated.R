options(pangoling.verbose = FALSE)





test_that("Deprecated functions issue warnings and return valid results", {
  skip_if_no_python_stuff()
  expect_warning(
    result <- masked_tokens_tbl("The apple doesn't fall far from the [MASK]"),
    "masked_tokens_pred_tbl"
  )
  expect_true(is.data.frame(result))  # Should return a dataframe
  
  expect_warning(
    result <- masked_lp(l_contexts = "The apple doesn't fall far from the", 
                        targets = "tree",
                        r_contexts = "."),
    "masked_targets_pred"
  )
  expect_true(is.numeric(result))  # Should return a numeric value
  
  expect_warning(
    result <- causal_next_tokens_tbl(context = "The apple doesn't fall far from the"),
    "causal_next_tokens_pred_tbl"
  )
  expect_true(is.data.frame(result))  # Should return a dataframe
  
  expect_warning(
    result <- causal_lp(x = c("The", "apple", "falls"), by = c(1,1,1)),
    "causal_targets_pred"
  )
  expect_true(is.numeric(result))  # Should return a numeric vector
  
  expect_warning(
    result <- causal_tokens_lp_tbl(texts = "The apple doesn't fall far from the tree."),
    "causal_tokens_pred_lst"
  )
  expect_true(is.data.frame(result))  # Should return a dataframe
  
  expect_warning(
    result <- causal_lp_mats(x = c("The", "apple", "falls"), 
                             by = c(1,1,1)),
    "causal_pred_mats"
  )
  expect_true(is.list(result))  # Should return a list of matrices
  
  # expect_warning(
  #   result <- char_to_token("tree", tkzr = NULL),
  #   "deprecated"
  # )
  # expect_true(is.list(result))
  
  # expect_warning(
  #   result <- num_to_token(1234, tkzr = NULL),
  #   "deprecated"
  # )
  # expect_true(is.character(result))
})
