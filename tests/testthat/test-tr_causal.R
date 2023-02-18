options(pangoling.verbose = FALSE)

prov <- "The apple doesn't fall far from the tree"
# realizes is differently encoded at the beginning or end
sent2 <- "He realizes something."
sent3 <- "realizes something."
prov_words <- strsplit(prov, " ")[[1]]
sent2_words <- strsplit(sent2, " ")[[1]]
sent3_words <- strsplit(sent3, " ")[[1]]

test_that("gpt2 load and gets config", {
  skip_if_no_python_stuff()
  expect_invisible(causal_preload())
  conf_lst <- causal_config()
  expect_true(is.list(conf_lst))
  expect_equal(
    conf_lst$`_name_or_path`,
    getOption("pangoling.causal.default")
  )
})

test_that("errors work", {
  skip_if_no_python_stuff()
  expect_error(causal_lp("It"))
  expect_error(causal_lp(c("It", "is."), .by = 3))
})

test_that("gpt2 get prob work", {
  skip_if_no_python_stuff()
  cont <-
    causal_next_tokens_tbl(context = "The apple doesn't fall far from the")
  expect_equal(sum(exp(cont$lp)), 1, tolerance = .0001)
  expect_equal(cont[1]$token, "Ġtree")
  lp_prov <- causal_lp(x = prov_words)
  expect_equal(names(lp_prov), prov_words)
  lp_sent2 <- causal_lp(x = sent2_words)
  expect_equal(names(lp_sent2), sent2_words)
  lp_sent3 <- causal_lp(x = sent3_words)
  expect_equal(names(lp_sent3), sent3_words)
  expect_equal(cont$lp[1], unname(lp_prov[[8]]), tolerance = .0001)
  lp_prov_mat <- causal_lp_mats(x = prov_words)
  mat <- lp_prov_mat[[1]]
  expect_equal(
    c(
      NA,
      mat["Ġapple", 2],
      mat["Ġdoesn", 3] + mat["'t", 4],
      mat["Ġfall", 5],
      mat["Ġfar", 6],
      mat["Ġfrom", 7],
      mat["Ġthe", 8],
      mat["Ġtree", 9]
    ),
    unname(unlist(lp_prov))
  )
  expect_equal(rownames(lp_prov_mat[[1]]), transformer_vocab())
  expect_equal(sum(exp(mat[, 2])), 1, tolerance = .0001) # sums to one

  # regex
  lp_prov2 <-
    causal_lp(
      x = strsplit(paste0(prov, "."), " ")[[1]],
      ignore_regex = "[[:punct:]]"
    )
  expect_equal(unname(lp_prov), unname(lp_prov2), tolerance = 0.001)

  ##
  sent <- "This is it, is it?"
  sent_w <- strsplit(sent, " ")[[1]]
  lp_sent <- causal_lp(x = sent_w)
  lp_sent2 <-
    causal_lp(x = sent_w, ignore_regex = "^[[:punct:]]$")
  expect_equal(lp_sent[c(-3, -5)], lp_sent2[c(-3, -5)])

  lp_sent_rep <-
    causal_lp(
      x = rep(sent_w, 2),
      .by = rep(seq_len(2), each = length(sent_w))
    )
  expect_equal(
    unname(lp_sent_rep[seq_along(sent_w)]),
    unname(lp_sent_rep[(length(sent_w) + 1):(2 * length(sent_w))])
  )


})

test_that("batches work", {
  skip_if_no_python_stuff()
  texts <- rep(c("This is not it.","This is it."),5)
  lp_batch <- causal_tokens_lp_tbl(texts,
                             batch_size =3, .id = ".id")

  lp_nobatch <- causal_tokens_lp_tbl(texts,
                             batch_size =1, .id = ".id")
  expect_equal(lp_batch, lp_nobatch, tolerance = .0001)
  df <- data.frame(x = rep(c(prov_words, sent2_words),3),
                   .id = c(rep(1, length(prov_words)),
                           rep(2, length(sent2_words)),
                           rep(3, length(prov_words)),
                           rep(4, length(sent2_words)),
                           rep(5, length(prov_words)),
                           rep(6, length(sent2_words))
                           ))
  lp_2_batch <- causal_lp(x = df$x, .by = df$.id, batch_size = 4)
  lp_2_no_batch <- causal_lp(x = df$x, .by = df$.id, batch_size = 1)
  expect_equal(lp_2_batch, lp_2_no_batch, tolerance = .0001)

})

test_that("can handle extra parameters", {
  skip_if_no_python_stuff()

  tkns <- tokenize_lst("This isn't it.")[[1]]
  token_lp <- causal_tokens_lp_tbl("This isn't it.")
  token_lp2 <- causal_tokens_lp_tbl(texts = "This isn't it.", add_special_tokens = TRUE)
  token_lp3 <- causal_tokens_lp_tbl(texts = "<|endoftext|>This isn't it.")
  expect_equal(token_lp$token, tkns)
  expect_equal(token_lp$token, token_lp2$token[-1])
  expect_equal(token_lp2, token_lp3)

  mat <- causal_lp_mats("This isn't it.")[[1]]
  expect_equal(
    token_lp$lp,
    tidytable::map_dbl(seq_along(tkns), ~ mat[token_lp$token[.x], .x])
  )
})

test_that("can handle extra parameters", {
  skip_if_no_python_stuff()
  probs <- causal_lp(x = c("This", "is", "it"), add_special_tokens = TRUE)
  word_1_prob <- causal_next_tokens_tbl("<|endoftext|>")
  prob1 <- word_1_prob[token == "This"]$lp
  names(prob1) <- "This"
  expect_equal(probs[1], prob1, tolerance = 0.0001)

  probs_F <- causal_lp(x = c("This", "is", "it"), add_special_tokens = FALSE)
  expect_true(is.na(probs_F[1]))
  word_2_prob <- causal_next_tokens_tbl("This")
  prob2 <- word_2_prob[token == "Ġis"]$lp
  names(prob2) <- "is"
  expect_equal(probs_F[2], prob2, tolerance = .0001)
})


if (0) {
  test_that("can handle longer than 1024 input", {
    num0 <- paste0(1:485, sep = ",")
    lp_num0 <- get_causal_lp(x = num0, model = "sshleifer/tiny-gpt2")

    num1 <- paste0(1:500, sep = ",")
    lp_num1 <- get_causal_lp(
      x = num1,
      model = "sshleifer/tiny-gpt2",
      stride = 10
    )
    expect_equal(lp_num0, lp_num1[1:485])
  })
}

test_that("other models using get prob don't fail", {
  skip_if_no_python_stuff()
  tokenize_lst("El bebé de cigüeña.", model = "flax-community/gpt-2-spanish")

  expect_no_error(causal_lp(
    x = c("El", "bebé", "de", "cigüeña."),
    model = "flax-community/gpt-2-spanish"
  ))

  expect_no_error(
    causal_lp(
      x = strsplit(paste0(prov, "."), " ")[[1]],
      model = "distilgpt2"
    )
  )
})
