
test_that("transformer models work", {
  cont <- get_tr_next_tokens_tbl("The apple doesn't fall far from the")
  expect_snapshot(cont)
  prov <- "The apple doesn't fall far from the tree"
  lp_prov <- get_tr_log_prob(x = strsplit(prov," ")[[1]], eot = 0)
  expect_snapshot(lp_prov)
  expect_equal(cont$log_prob[1], unname(lp_prov[8]), tolerance = .0001)
  lp_prov_mat <- get_tr_log_prob_mat(x = strsplit(prov," ")[[1]], eot = 0)
  mat <- lp_prov_mat[[1]]
  expect_equal(c(NA,mat["Ġapple",2], mat["Ġdoesn", 3]+mat["'t", 4],
    mat["Ġfall",5], mat["Ġfar",6], mat["Ġfrom",7], mat["Ġthe", 8], mat["Ġtree", 9]),
    unname(lp_prov))
  expect_equal(rownames(lp_prov_mat[[1]]), get_tr_vocab())
  expect_equal(sum(exp(mat[,2])), 1, tolerance = .0001) #sums to one
  lp_prov2 <- get_tr_log_prob(x = strsplit(paste0(prov,".")," ")[[1]], eot = 0)
  expect_snapshot(lp_prov2)
  lp_prov3 <- get_tr_log_prob(x = strsplit(paste0(prov,".")," ")[[1]], eot = 0, ignore_regex = "[[:punct:]]")
  expect_equal(unname(lp_prov), unname(lp_prov3), tolerance = 0.001)

  lp_provd<- get_tr_log_prob(x = strsplit(paste0(prov,".")," ")[[1]], eot = 0, model = "distilgpt2")
  expect_snapshot(lp_provd)

  ent_prov <-get_tr_entropy(x = strsplit(prov," ")[[1]], eot = 0)
  expect_equal(c(NA,-sum(exp(mat[,2])*mat[,2]),
                 -sum(exp(mat[,3])*mat[,3])+-sum(exp(mat[,4])*mat[,4]),
                 -sum(exp(mat[,5])*mat[,5]),
                 -sum(exp(mat[,6])*mat[,6]),
                 -sum(exp(mat[,7])*mat[,7]),
                 -sum(exp(mat[,8])*mat[,8]),
                 -sum(exp(mat[,9])*mat[,9])),
               unname(ent_prov))
})

