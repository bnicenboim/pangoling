

test_that("gp2 get prob work", {
  cont <-
    get_causal_next_tokens_tbl("The apple doesn't fall far from the")
  expect_snapshot(cont)
  prov <- "The apple doesn't fall far from the tree"
  prov_words <- strsplit(prov, " ")[[1]]
  lp_prov <- get_causal_log_prob(x = prov_words, eot = 0)
  expect_snapshot(lp_prov)
  expect_equal(cont$log_prob[1], unname(lp_prov[[8]]), tolerance = .0001)
  lp_prov_mat <- get_causal_log_prob_mat(x = prov_words, eot = 0)
  mat <- lp_prov_mat[[1]][[1]]
  expect_equal(c(NA, mat["Ġapple", 2], mat["Ġdoesn", 3] + mat["'t", 4],
                 mat["Ġfall", 5], mat["Ġfar", 6], mat["Ġfrom", 7], mat["Ġthe", 8], mat["Ġtree", 9]),
               unname(unlist(lp_prov)))
  expect_equal(rownames(lp_prov_mat[[1]][[1]]), get_tr_vocab())
  expect_equal(sum(exp(mat[, 2])), 1, tolerance = .0001) #sums to one
  lp_prov2 <-
    get_causal_log_prob(x = strsplit(paste0(prov, "."), " ")[[1]], eot = 0)
  expect_snapshot(lp_prov2)
  lp_prov3 <-
    get_causal_log_prob(x = strsplit(paste0(prov, "."), " ")[[1]],
                        eot = 0,
                        ignore_regex = "[[:punct:]]")
  expect_equal(unname(lp_prov), unname(lp_prov3), tolerance = 0.001)
  sent <- "This is it, is it?"
  sent_w <- strsplit(sent, " ")[[1]]
  lp_sent <- get_causal_log_prob(x = sent_w)
  lp_sent2 <-
    get_causal_log_prob(x = sent_w, ignore_regex = "^[[:punct:]]$")
  expect_equal(lp_sent[c(-3, -5)], lp_sent2[c(-3, -5)])
  lp_sent_rep <-
    get_causal_log_prob(
      x = rep(sent_w, 2),
      by = rep(1:2, each = length(sent_w)),
      eot = 0
    )
  expect_equal(unname(lp_sent_rep[1:length(sent_w)]),
               unname(lp_sent_rep[(length(sent_w) + 1):(2 * length(sent_w))]))
  lp_sent_rep_j <- get_causal_log_prob(x = rep(sent_w, 2), eot = 0)
})

test_that("other models using get prob work", {
  lp_provd <-
    get_causal_log_prob(x = strsplit(paste0(prov, "."), " ")[[1]],
                        eot = 0,
                        model = "distilgpt2")
  expect_snapshot(lp_provd)
})
# cont <- get_tr_next_tokens_tbl("The apple doesn't fall far from the",model = "distilbert-base-uncased")

test_that("masked models work", {
  sent <- "This is it, is it?"
  sent_w <- strsplit(sent, " ")[[1]]
  lp_sent_rep_m <-
    get_masked_log_prob(x = rep(sent_w, 2), by = rep(1:2, each = length(sent_w)))
  expect_equal(unname(lp_sent_rep_m[1:length(sent_w)]),
               unname(lp_sent_rep_m[(length(sent_w) + 1):(2 * length(sent_w))]))
  # masked_sentence <- "The apple doesn't fall far from the [MASK]."
  # pr_mask <- get_masked_tokens_tbl(masked_sentence)
  # expect_snapshot(pr_mask)
  masked_sentence2 <- "The apple doesn't fall far from [MASK] tree."
  pr_mask2 <- get_masked_tokens_tbl(masked_sentence2)
  expect_snapshot(pr_mask2)
  masked_sentence3 <- "This is [MASK] [MASK] cat."
  pr_mask3 <- get_masked_tokens_tbl(masked_sentence3)
  expect_snapshot(pr_mask3)

  lp_wbw <- get_masked_log_prob(x = c("This","isn't","a","dream."))
  lp_1 <- get_masked_tokens_tbl("[MASK][MASK][MASK][MASK][MASK] [MASK][MASK]" )
  lp_2 <- get_masked_tokens_tbl("This [MASK][MASK][MASK][MASK] [MASK][MASK]" )
  lp_3 <- get_masked_tokens_tbl("This isn't [MASK] [MASK][MASK]" )

  expect_equal(lp_wbw[[1]][[1]], lp_1[mask_n==1 & token == "this",]$log_prob)
  expect_equal(lp_wbw[[2]][[2]],
               lp_1[mask_n==2 & token == "isn",]$log_prob +
                lp_1[mask_n==3 & token == "'",]$log_prob+
               lp_1[mask_n==4 & token == "t",]$log_prob, tolerance =  0.01)
  lp_wbw[[2]]
  lp_2[mask_n==1 & token == "isn",]$log_prob +
    lp_2[mask_n==2 & token == "'",]$log_prob+
    lp_2[mask_n==3 & token == "t",]$log_prob

  lp_3 <- get_masked_tokens_tbl("This isn't a [MASK][MASK]" )
  dream. <- lp_3[mask_n==1 & token =="dream",]$log_prob + lp_3[mask_n==2 & token ==".",]$log_prob
  expect_equal(lp_wbw$`1. dream.`[1], dream.)
})

lp[[1]][get_tr_vocab(model)=="isn", "isn"]+
lp[[1]][get_tr_vocab(model)=="'", "'"]+
lp[[1]][get_tr_vocab(model)=="t", "t"]
isn <- out_lm$logits[0][2]
a <- out_lm$logits[0][3]
tt <- out_lm$logits[0][4]
lsm2 <- reticulate::py_to_r(torch$log_softmax(isn, dim = -1L)$tolist())
lsm3 <- reticulate::py_to_r(torch$log_softmax(a, dim = -1L)$tolist())
lsm4 <- reticulate::py_to_r(torch$log_softmax(tt, dim = -1L)$tolist())
lsm2[get_tr_vocab(model)=="isn"] + lsm3[get_tr_vocab(model)=="'"] +
lsm4[get_tr_vocab(model)=="t"]

test_that("entropy works", {
  ent_prov <- get_causal_entropy(x = strsplit(prov, " ")[[1]], eot = 0)
  expect_equal(c(
    NA,
    -sum(exp(mat[, 2]) * mat[, 2]),-sum(exp(mat[, 3]) * mat[, 3])+-sum(exp(mat[, 4]) *
                                                                         mat[, 4]),-sum(exp(mat[, 5]) * mat[, 5]),-sum(exp(mat[, 6]) * mat[, 6]),-sum(exp(mat[, 7]) *
                                                                                                                                                        mat[, 7]),-sum(exp(mat[, 8]) * mat[, 8]),-sum(exp(mat[, 9]) * mat[, 9])
  ),
  unname(ent_prov))
})
