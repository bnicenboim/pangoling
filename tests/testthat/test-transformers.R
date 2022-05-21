
test_that("transformer models work", {
  cont <- get_tr_next_tokens_tbl("The apple doesn't fall far from the")
  expect_snapshot(cont)
  prov <- "The apple doesn't fall far from the tree"
  lp_prov <- get_tr_log_prob(x = strsplit(prov," ")[[1]], eot = 0)
  expect_snapshot(lp_prov)
  expect_equal(cont$log_prob[1], unname(lp_prov[8]), tolerance = .0001)
  })

