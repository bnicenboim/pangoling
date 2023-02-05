#'
#' @param context Context
#' @param model Name of a pretrained model stored on the huggingface.co. (Maybe a path to a  model (.pt or .bin file) stored locally will work.)
#'
#' @return
#'
get_masked_tokens_tbl <- function(masked_sentence, model = "distilbert-base-uncased", add_special_tokens = TRUE, config_model = NULL, config_tokenizer = NULL) {
  message_verbose("Processing using masked model '", model, "'...")
  tkzr <- tokenizer(model, config = config_tokenizer)
  masked_tensor <- tkzr$encode(masked_sentence, return_tensors = "pt", add_special_tokens = add_special_tokens)
  outputs <- lang_model(model, task = "masked", config = config_model)(masked_tensor)
  mask_pos <- which(masked_tensor$tolist()[[1]] == tkzr$mask_token_id)
  logits_masks <- outputs$logits[0][mask_pos - 1] # python starts in 0
  lp <- reticulate::py_to_r(torch$log_softmax(logits_masks, dim = -1L)$tolist())
  if (length(mask_pos) == 1) lp <- list(lp) # to keep it consistent
  names(lp) <- paste0("mask_", 1:length(lp))
  vocab <- sort(unlist(tkzr$get_vocab())) |> names()
  lp |> tidytable::map_dfr.(~
    tidytable::tidytable(token = vocab, log_prob = .x) |>
      tidytable::arrange.(-log_prob), .id = "mask_n")
}

#'
#'
#'
#' #' Get the log probability of each word phrase of a vector given its previous context using a transformer model from huggingface.co.
#' #'
#' #' In case of errors check the status of https://status.huggingface.co/
#' #'
#' #' @param x Vector of words, phrases or texts.
#' #' @param by Vector that indicates how the text should be split.
#' #' @param model Name of a pretrained model stored on the huggingface.co. (Maybe a path to a  model (.pt or .bin file) stored locally will work.)
#' #' @ignore_regex Can ignore certain characters when calculates the log probabilities. For example `^[[:punct:]]$` will ignore all punctuation  that stands alone in a token.
#' #' @eot 0 Does nothing, 1 adds an end of text tag at the beginning of a vector, 2 adds an end of text tag and a space at the beginning of a vector.
#' #'
#' #' @return a vector of log probabilities.
#' #'
#' #' @noRd
#' get_masked_log_prob <- function(x,
#'                                 by = rep(1, length(x)),
#'                                 ignore_regex = "",
#'                                 model = "distilbert-base-uncased",
#'                                 max_batch_size = 50,
#'                                 window_stride = 5,
#'                                 n_plus = 0,
#'                                 max_tokens = max_tokens_masked(model)) {
#'   get_lm_lp(x =x,
#'             by=by,
#'             ignore_regex = ignore_regex,
#'             type = "masked",
#'             model = model,
#'             n_plus = n_plus,
#'             max_batch_size = max_batch_size)
#' }
#'
#'
#' #' Title
#' #'
#' #' @param x
#' #' @param by
#' #' @param eot
#' #' @param model
#' #'
#' #' @return
#' #'
#' #' @noRd
#' get_masked_log_prob_mat <- function(x,
#'                                     by = rep(1, length(x)),
#'                                     model = "distilbert-base-uncased",
#'                                     max_batch_size = 50,
#'                                     window_stride = 5,
#'                                     n_plus = 3,
#'                                     max_tokens = max_tokens_masked(model)) {
#'   x <- trimws(x, whitespace = "[ \t]")
#'   texts <- split(x, by)
#'   N <- length(texts)
#'   tidytable::map2.(texts,names(texts), function(words,item) {
#'     masked_log_prob_mat(words,
#'                         model = model,
#'                         max_batch_size =max_batch_size,
#'                         window_stride = window_stride,
#'                         n_plus = n_plus,
#'                         max_tokens = max_tokens)
#'
#'   })
#' }
#'
#' #' @noRd
#' masked_log_prob_mat <- function(x,
#'                                 model = "distilbert-base-uncased",
#'                                 max_batch_size = 50,
#'                                 window_stride = "auto",
#'                                 n_plus = 3,
#'                                 max_tokens = max_tokens_masked(model)){
#'   # n_before = 1 -> regular prediction to n+1
#'   text <-  paste0(x, collapse = " ")
#'   input_ids  <-
#'     reticulate::py_to_r(tokenizer(model)(text, return_tensors = "pt")$input_ids$tolist())[[1]]
#'   tensor_size <- length(input_ids)
#'
#'
#'   if(tensor_size > max_tokens){
#'     if(is.numeric(window_stride)){
#'     input_ids_lst <- slide_tokens(input_ids, max_tokens, window_stride)
#'     # where the original words are
#'     rel_pos_w <- rel_pos_slide(input_ids, max_tokens, window_stride)
#'     # reconstructs input_ids
#'     input_idsr <- unlist(tidytable::map2.(input_ids_lst, rel_pos_w, ~.x[.y]))
#'     stopifnot(input_ids ==input_idsr)
#'     ## adds predictions based on n_plus
#'     ## where the masks should be:
#'     rel_pos <- tidytable::map.(rel_pos_w, ~ seq.int(pmax(min(.x-(n_plus-1)),1), max(.x)))
#'     }
#'   } else {
#'     input_ids_lst <- list(input_ids)
#'     rel_pos <- list(seq_along(input_ids))
#'     rel_pos_w <- rel_pos
#'   }
#'
#'   if(n_plus==0) n_plus <- max(unlist(lapply(input_ids_lst, length)))-1
#'   mask_id <- reticulate::py_to_r(tokenizer(model)$mask_token_id)
#'   special_ids <- reticulate::py_to_r(tokenizer(model)$all_special_ids)
#'
#'   # masks ids in the unlisted sequential input_id
#'   masks_id <- which(!input_ids %in% special_ids)
#'   # number of masks that need to be applied and where
#'   # if there are several windows, in the first is everywhere except special ids (beggining), afterwards it's only on the new tokens (n_plus =1), or as (n_plus-1) before (indexes from 1)
#'   # masks <- c(list(which(!input_ids_lst[[1]] %in% special_ids))
#'   #   ,lapply(input_ids_lst[-1], function(i) which(i %in% i[(length(i)-stride- n_plus + 2):length(i)] & !i %in% special_ids)))
#'   #
#'   masks <- tidytable::map2.(input_ids_lst, rel_pos, ~ {
#'     which(!.x %in% special_ids & seq_along(.x) %in% .y)
#'   })
#'
#'   # n_input_ids_lst <- purrr::map2(input_ids_lst,nmasks, function(i,n) rep(list(i), n))
#'
#'   # input_ids_masked_lst <- tidytable::map2.(n_input_ids_lst, masks,
#'   #                                          function(i,m)  tidytable::map2.(i,seq_along(m), ~ {
#'   #   .x[m[.y:length(m)]] <- mask_id
#'   #   .x
#'   # }))
#'
#'   ## output masks from n_plus -1 of the word to be predicted, one by one:
#'   input_ids_masked_lst <- tidytable::map2.(input_ids_lst, masks,
#'         function(i,m){
#'           # i <- input_ids_lst[[2]]
#'           # m <- masks[[2]]
#'         m_pos <-   rev(tidytable::map.((seq_along(m)-1), ~ rev(m[length(m):(length(m)-.x)])))
#'         tidytable::map.(m_pos, ~ {i[.x] <- mask_id
#'         i})
#'
#'         }  )
#'
#'   #makes a tensor for the language model
#'   tinput_ids_masked_lst <- lapply(input_ids_masked_lst, torch$tensor)
#'   # Is there a batch potentially too large?
#'   n_tensors <- lapply(tinput_ids_masked_lst, function(t) reticulate::py_to_r(t$shape[0]))
#'   # split them in groups of this size
#'   n_groups <- lapply(n_tensors, function(n) n / max_batch_size)
#'
#'   tensor_groups <- lapply(n_tensors, function(n) split(0:(n-1), ceiling((1:n)/max_batch_size)))
#'
#'  # new tensors respecting the maximum size:
#'   tinput_ids_masked_lst_g <- tidytable::map2.(tinput_ids_masked_lst,tensor_groups, function(tlst, tg) {
#'     lapply(tg, function(i) {
#'
#'       if(length(reticulate::py_to_r(tlst[i]$shape))==1){
#'         torch$u(tlst[i],1)
#'       } else {
#'         tlst[i]
#'       }
#'     })
#'     })
#'   #tlst[0]$unsqueeze(0L)$shape
#'
#'   message_verbose("Processing ", sum(unlist(n_tensors))," tensors in ", length(unlist(tinput_ids_masked_lst_g))," groups of (maximum) ",max(sapply(tensor_groups, length)) , " batches.")
#'   message_verbose("Processing using masked model '", model,"'...")
#'
#'   #out_lm <- lang_model(model, task = "masked")(tinput_ids_masked)
#'   out_lm_lst_g <- lapply(tinput_ids_masked_lst_g, function(g) lapply(g, function(t) lang_model(model, task = "masked")(t)$logits))
#'
#'
#'   # first reorganize if batches were split by creating tensors
#'   out_lm_lst <-  lapply(out_lm_lst_g, function(g) torch$row_stack(unname(g)))
#'
#'   tokens <- reticulate::py_to_r(tokenizer(model)$convert_ids_to_tokens(input_ids[masks_id]))
#'   vocab <- get_tr_vocab(model)
#'
#'   out_lm_lstl <- tidytable::map.(out_lm_lst, ~ reticulate::py_to_r(.x$tolist()))
#'   # length(out_lm_lstl[[1]][[1]][[1]])
#'   #  out_lm_lst[[1]]$shape
#' #   if(0){
#' #    probs <- torch$log_softmax(out_lm_lst[[1]], dim = -1L)
#' #    probs <- reticulate::py_to_r(probs$tolist())
#' #    probs[[1]][[3]][vocab=="isn"]
#'   #probs[[2]][[2]][vocab=="isn"]
#'   #probs[[1]][[2]][vocab=="this"]
#' # }
#'   # #
#'   # length(masks[[1]])
#'   # length(out_lm_lstl[[1]])
#'   lp <- tidytable::map.(seq.int(n_plus)-1, function(s){
#'
#'
#'   lp_g <- tidytable::pmap.(list(rel_pos_w, masks,out_lm_lstl),
#'                    function(rel_pos_w_g, masks_g, out_lm_lstl_g)
#'                    {
#'   #rel_pos_w_g <- rel_pos_w[[1]]
#'   #masks_g <- masks[[1]]
#'   #out_lm_lstl_g <- out_lm_lstl[[1]]
#'   tidytable::map.(rel_pos_w_g, function(w){
#'     #w <- rel_pos_w_g[[1]]
#'     i <- which(masks_g == w) - s
#'     w_pos <- masks_g[masks_g == w]
#'     if(length(i)==0 || i < 1){
#'       rep(NA, length(vocab))
#'     } else {
#'       log_softmax(out_lm_lstl_g[[i]][[w_pos]])
#'     }
#'   })
#'                    })
#'   lp <- unlist(lp_g,recursive = FALSE)
#'   stopifnot(length(lp) ==length(input_ids))
#'   lp_mat <- matrix(unlist(lp), nrow = length(vocab))
#'   rownames(lp_mat) <- vocab
#'   colnames(lp_mat) <-  get_tokens(input_ids, model)
#'   lp_mat <- lp_mat[,colnames(lp_mat) %in% tokens]
#'   lp_mat
#'   })
#'   lp
#'   #
#' # final_tokens <- tidytable::map2.(out_lm_lst[-1],rel_pos_slide(input_ids, max_tokens, stride), function(l,pos) {
#' #      len <- reticulate::py_to_r(l$shape[1])
#' #      dim0 <- reticulate::py_to_r(l$shape[0])
#' #      # probably not the most efficient way to get the last tokens of the 2 dimension:
#' #      torch$stack(lapply(0:(dim0-1), function(i) l[i][(len-pos):(len-1)]))
#' #    })
#' #
#' # logits <- torch$row_stack(c(out_lm_lst[[1]],final_tokens ))
#' # #
#' #
#' #
#' #
#' #
#' #   # python objects below, indexes need to start from 0
#' #   # .x is the batch index, iterates over the input_ids_masked
#' #   # .y in the masked word position
#' #   # lp <- tidytable::map2.(0:(nmasks-1),(masks-1), ~ reticulate::py_to_r(torch$log_softmax(out_lm$logits[.x][.y], dim = -1L))$tolist())
#' #
#' #     lp <- tidytable::pmap.(list(out_lm_lst, nmasks,masks), function(out_lm, nm,m) {
#' #   #lp <-
#' #     lapply(1:nm, function(n)
#' #   { # n-1 indexes the masked sentence (Starts from 0)
#' #     #m-1 are the indexes of the masks in the sentences (starts from 0), takes the last n:nm masks
#' #     logits <- out_lm[n-1][(m-1)[n:nm]]
#' #     lsm <- torch$log_softmax(logits, dim = -1L)
#' #     mat_mask <- reticulate::py_to_r(lsm)$tolist() |>
#' #       unlist() |>
#' #       matrix(ncol =length(n:nm))
#' #     mat_mask <- cbind(matrix(NA, nrow = nrow(mat_mask), ncol = n-1), mat_mask)
#' #     rownames(mat_mask)  <- get_tr_vocab(model)
#' #     #colnames(mat_mask) <- unlist(tokens)
#' #     mat_mask
#' #   }
#' #   )
#' #   })
#' #     lapply(lp)
#' #   # stores the predictions from wordn to wordn+1..N in each list
#' #   lp
#'   # this below doesn't seem to be right
#'   # lp_by_pred <- lapply(0:(nmasks-1),   function(m){
#'   #   cbind(matrix(NA, nrow = nrow(lp[[1]]), ncol = m),
#'   #         lapply(1:(nmasks-m), function(n){
#'   #           lp[[n]][,n+m,drop = FALSE]
#'   #         }) |> do.call("cbind",.))
#'   # })
#'   #  lp_by_pred
#' }
#'
#' ###' @noRd
#' #' get_log_prior <- function(x, model = "distilbert-base-uncased") {
#' #'
#' #'   #models using uncased wikipedia and bookcorpus
#' #'   model_wiki_bookcorpus <- c("distilbert-base-uncased", "bert-base-uncased", "bert-large-uncased")
#' #'   if(!model %in% model_wiki_bookcorpus) {
#' #'     stop2("Only the following uncased models based on wikipedia and bookcorpus are supported for now: ", paste(model_wiki_bookcorpus, collapse =", "))
#' #'   }
#' #'   tokens <- get_tokens(x = x, model = model)
#' #'   tidytable::map_dbl.(tokens, function(t) {
#' #'     t <- tokens[[3]]
#' #'     df_bert_unigram |>
#' #'       tidytable::filter.(unigram == x[[3]]) |>
#' #'       tidytable::pull.(lprior)
#' #'     sum(tidytable::map_dbl.(t, ~ df_bert_unigram |>
#' #'                               tidytable::filter.(unigram == .x) |>
#' #'                               tidytable::pull.(lprior)))
#' #' })
#' #'   ## another attemped based on looking at the lp of everyword in a completely masked sentence
#' #'   # mask_token_id <- reticulate::py_to_r(tokenizer(model)$mask_token_id)
#' #'   # mask_tensor <- torch$tensor(rep(mask_token_id,512))$unsqueeze(0L)
#' #'   mask_tensor <-
#' #'   tokenizer(model)(paste0(rep("[MASK]",510), collapse=""), return_tensors = "pt")$input_ids
#' #'    vocab_l <-  lang_model(model, task = "masked")(mask_tensor)$logits
#' #'   vocab_l <- vocab_l$tolist()
#' #'   vocab <- get_tr_vocab(model)
#' #'   vocab_lp <- lapply(vocab_l[[1]], log_softmax)
#' #'   vocab_lp_mat <- unlist(vocab_lp) |>
#' #'           matrix(nrow =length(vocab))
#' #'   #dim(vocab_lp_mat)
#' #'   vocab_lp <- rowSums(vocab_lp_mat[,2:511])
#' #'
#' #'   names(vocab_lp) <- vocab
#' #'   vocab_lp[names(vocab_lp) =="."]
#' #'   exp(vocab_lp[names(vocab_lp) =="a"])
#' #'   df_freq <- LexOPS::lexops |> select.(string, fpmw.SUBTLEX_US)
#' #'   df_bfreq <- tidytable(string = names(vocab_lp), lp = vocab_lp)
#' #'   df_f <- left_join.(df_freq, df_bfreq)
#' #'   cor(log(df_f$fpmw.SUBTLEX_US), df_f$lp,"complete")
#' #'   df2 <- left_join.(df_bert_unigram, rename.(df_bfreq, unigram= string))
#' #'   cor(df2$lprior,df2$lp, "complete")
#' #' }
#' #' #
#'
#' #' Title
#' #'
#' #' @param model
#' #'
#' #' @return
#' #'
#' #' @examples
#' #' @noRd
#' max_tokens_masked <- function(model = "distilbert-base-uncased"){
#'   lang_model(model, task = "masked")$config$max_position_embeddings
#' }
#'
#'
