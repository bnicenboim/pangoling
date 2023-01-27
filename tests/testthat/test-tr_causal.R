
test_that("gp2 get prob work", {
  cont <-
    get_causal_next_tokens_tbl("The apple doesn't fall far from the")
  expect_snapshot(cont)
  prov <- "The apple doesn't fall far from the tree"
  prov_words <- strsplit(prov, " ")[[1]]
  lp_prov <- get_causal_log_prob(x = prov_words)
  expect_equal(names(lp_prov), paste0("1.", prov_words))
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

  expect_snapshot(
    get_causal_log_prob(x = c("This","is","it"),add_bos_token = TRUE)
  )

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

get_tokens("El bebé de cigüeña.", model = "flax-community/gpt-2-spanish")
  expect_snapshot(
    get_causal_log_prob(x = c("El","bebé","de" ,"cigüeña."), model = "flax-community/gpt-2-spanish")
  )
pangoling:::causal_log_prob_mat(c("El","bebé","de" ,"cigüeña."), model = "flax-community/gpt-2-spanish")

  "AttributeError: 'GPT2TokenizerFast' object has no attribute 'encode'"


  lp_provd <-
    get_causal_log_prob(x = strsplit(paste0(prov, "."), " ")[[1]],
                        model = "distilgpt2")
  expect_snapshot(lp_provd)
})

# cont <- get_tr_next_tokens_tbl("The apple doesn't fall far from the",model = "distilbert-base-uncased")

# test_that("masked models work", {
#   gets_0 <- get_masked_tokens_tbl(masked_sentence = "This isn't  [MASK]", model ="prajjwal1/bert-tiny" )
#   gets_1 <- get_masked_tokens_tbl("[CLS] This isn't  [MASK] [SEP]", model ="prajjwal1/bert-tiny" ,add_special_tokens = FALSE)
#   expect_equal(gets_0,gets_1)
#
#
#   sent <- "This is it, is it?"
#   sent_w <- strsplit(sent, " ")[[1]]
#   lp_sent_rep_m <-
#     get_masked_log_prob(x = rep(sent_w, 2), by = rep(1:2, each = length(sent_w)))
#   expect_equal(unname(lp_sent_rep_m[1:length(sent_w)]),
#                unname(lp_sent_rep_m[(length(sent_w) + 1):(2 * length(sent_w))]))
#   # masked_sentence <- "The apple doesn't fall far from the [MASK]."
#   # pr_mask <- get_masked_tokens_tbl(masked_sentence)
#   # expect_snapshot(pr_mask)
#   masked_sentence2 <- "The apple doesn't fall far from [MASK] tree."
#   pr_mask2 <- get_masked_tokens_tbl(masked_sentence2)
#   expect_snapshot(pr_mask2)
#   masked_sentence3 <- "This is [MASK] [MASK] cat."
#   pr_mask3 <- get_masked_tokens_tbl(masked_sentence3)
#   expect_snapshot(pr_mask3)
#
#   lp_wbw <- get_masked_log_prob(x = c("This","isn't","a","dream."), n_plus = 0)
#   lp_1 <- get_masked_tokens_tbl("[MASK][MASK][MASK][MASK][MASK] [MASK][MASK]" )
#   lp_2 <- get_masked_tokens_tbl("This [MASK][MASK][MASK][MASK] [MASK][MASK]" )
#   lp_3 <- get_masked_tokens_tbl("This isn[MASK][MASK][MASK] [MASK][MASK]" )
#   lp_4 <- get_masked_tokens_tbl("This isn'[MASK] [MASK][MASK]" )
#   lp_5 <- get_masked_tokens_tbl("This isn't [MASK] [MASK][MASK]" )
#   lp_6 <- get_masked_tokens_tbl("This isn't a [MASK][MASK]" )
#   lp_7 <- get_masked_tokens_tbl("This isn't a dream[MASK]" )
#
#   expect_equal(lp_wbw[[1]][[1]], lp_1[mask_n==1 & token == "this",]$log_prob, tolerance =  0.0001)
#   expect_equal(lp_wbw[[2]][[1]],
#                lp_2[mask_n==1 & token == "isn",]$log_prob +
#                  lp_3[mask_n==1 & token == "'",]$log_prob+
#                  lp_4[mask_n==1 & token == "t",]$log_prob, tolerance = 0.001)
#
#   expect_equal(lp_wbw[[3]][[1]], lp_5[mask_n==1 & token == "a",]$log_prob, tolerance =  0.01)
#   dream. <- lp_6[mask_n==1 & token =="dream",]$log_prob + lp_7[mask_n==1 & token ==".",]$log_prob
#   expect_equal(lp_wbw$`1. dream.`[1], dream., tolerance = 0.0001)
#
#   #n + 2 pred
#
#   expect_equal(lp_wbw[[2]][[2]],
#                lp_1[mask_n==2 & token == "isn",]$log_prob +
#                  lp_2[mask_n==2 & token == "'",]$log_prob+
#                  lp_3[mask_n==2 & token == "t",]$log_prob, tolerance =  0.01)
#   #NOTICE HERE, IMPORTANT TO EXPLAIN
#   expect_equal(lp_wbw[[3]][[2]],
#                lp_2[mask_n==4 & token == "a",]$log_prob, tolerance =  0.01)
#
#
#   ###
#   num0 <- paste0(1:30, sep=",")
#   ntokens(paste(num0, collapse =" "), model ="prajjwal1/bert-tiny")
#   lp_num0_all <- get_masked_log_prob(x = num0, model ="prajjwal1/bert-tiny", max_batch_size = 100)
#   lp_num0_g <- get_masked_log_prob(x = num0, model ="prajjwal1/bert-tiny", max_batch_size = 23)
#   expect_equal(lp_num0_all,lp_num0_g)
#
#
#
# })

