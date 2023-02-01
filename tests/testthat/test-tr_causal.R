options(pangoling.verbose = FALSE)
test_that("gp2 get prob work", {
  skip_if_no_python_stuff()
    cont <-
    get_causal_next_tokens_tbl("The apple doesn't fall far from the")
  expect_snapshot(cont)
  expect_equal(cont[1]$token, "Ġtree")
  prov <- "The apple doesn't fall far from the tree"
  prov_words <- strsplit(prov, " ")[[1]]
  lp_prov <- get_causal_log_prob(x = prov_words)
  #expect_equal(names(lp_prov), paste0("1.", prov_words))
  expect_equal(names(lp_prov),  prov_words)
  expect_snapshot(lp_prov)
  expect_equal(cont$log_prob[1], unname(lp_prov[[8]]), tolerance = .0001)
  lp_prov_mat <- get_causal_log_prob_mat(x = prov_words)
  mat <- lp_prov_mat[[1]][[1]]
  expect_equal(c(NA, mat["Ġapple", 2], mat["Ġdoesn", 3] + mat["'t", 4],
                 mat["Ġfall", 5], mat["Ġfar", 6], mat["Ġfrom", 7], mat["Ġthe", 8], mat["Ġtree", 9]),
               unname(unlist(lp_prov)))
  expect_equal(rownames(lp_prov_mat[[1]][[1]]), get_tr_vocab())
  expect_equal(sum(exp(mat[, 2])), 1, tolerance = .0001) #sums to one

  lp_prov2 <-
    get_causal_log_prob(x = strsplit(paste0(prov, "."), " ")[[1]])
  expect_snapshot(lp_prov2)
  #regex
  lp_prov3 <-
    get_causal_log_prob(x = strsplit(paste0(prov, "."), " ")[[1]],
                        ignore_regex = "[[:punct:]]")
  expect_equal(unname(lp_prov), unname(lp_prov3), tolerance = 0.001)

  ##
  sent <- "This is it, is it?"
  sent_w <- strsplit(sent, " ")[[1]]
  lp_sent <- get_causal_log_prob(x = sent_w)
  lp_sent2 <-
    get_causal_log_prob(x = sent_w, ignore_regex = "^[[:punct:]]$")
  expect_equal(lp_sent[c(-3, -5)], lp_sent2[c(-3, -5)])

  lp_sent_rep <-
    get_causal_log_prob(
      x = rep(sent_w, 2),
      .by = rep(1:2, each = length(sent_w))
    )
  expect_equal(unname(lp_sent_rep[1:length(sent_w)]),
               unname(lp_sent_rep[(length(sent_w) + 1):(2 * length(sent_w))]))
  lp_sent_rep_j <- get_causal_log_prob(x = rep(sent_w, 2))


})


test_that("can handle extra parameters",{
  skip_if_no_python_stuff()
  probs <- get_causal_log_prob(x = c("This","is","it"),add_bos_token = TRUE)
  expect_snapshot(probs)
  expect_true(!is.na(probs[1]))
  })


if(0){
test_that("can handle longer than 1024 input",{
  num0 <- paste0(1:485, sep=",")
  lp_num0 <- get_causal_log_prob(x = num0, model ="sshleifer/tiny-gpt2")

  num1 <- paste0(1:500, sep=",")
  lp_num1 <- get_causal_log_prob(x = num1, model = "sshleifer/tiny-gpt2", stride = 10)
  expect_equal(lp_num0, lp_num1[1:485])
})
}

test_that("other models using get prob work", {
  skip_if_no_python_stuff()
get_tokens("El bebé de cigüeña.", model = "flax-community/gpt-2-spanish")
if(0){ #not working :()
    expect_snapshot(
    get_causal_log_prob(x = c("El","bebé","de" ,"cigüeña."), model = "flax-community/gpt-2-spanish")
  )
}

  lp_provd <-
    get_causal_log_prob(x = strsplit(paste0(prov, "."), " ")[[1]],
                        model = "distilgpt2")
  expect_snapshot(lp_provd)
})

