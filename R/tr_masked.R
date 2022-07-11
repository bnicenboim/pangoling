#' Title
#'
#' @param model
#'
#' @return
#' @export
#'
#' @examples
max_tokens_masked <- function(model = "distilbert-base-uncased"){
  lang_model(model, task = "masked")$config$max_position_embeddings
}


#'
#' @param context Context
#' @param model Name of a pretrained model stored on the huggingface.co. (Maybe a path to a  model (.pt or .bin file) stored locally will work.)
#'
#' @return
#'
#' @examples
#' @export
get_masked_tokens_tbl <- function(masked_sentence, model = "distilbert-base-uncased") {
  masked_tensor <-
    tokenizer(model)(masked_sentence, return_tensors = "pt")$input_ids

  outputs <- lang_model(model, "masked")(masked_tensor)
  mask_pos <-  which(reticulate::py_to_r(masked_tensor$tolist()[0])== reticulate::py_to_r(tokenizer(model)$mask_token_id))
  logits_masks <- outputs$logits[0][mask_pos-1] # python starts in 0
  lp <- reticulate::py_to_r(torch$log_softmax(logits_masks, dim = -1L)$tolist())
  if(length(mask_pos)==1) lp <- list(lp) #to keep it consistent
  #names(lp) <- paste0("mask_",1:length(lp))
    lp |> tidytable::map_dfr.(~ tidytable::tidytable(token = get_tr_vocab(model), log_prob = .x) |>
                                 tidytable::arrange.(-log_prob), .id = "mask_n" )
    #   as_tidytable() |>
    #   tidytable::mutate.(token = get_tr_vocab(model), .before = tidyselect::everything()) |>
    #   tidytable
    # tidytable::arrange.(-mask_1)
}



#' Get the log probability of each word phrase of a vector given its previous context using a transformer model from huggingface.co/.
#'
#' In case of errors check the status of https://status.huggingface.co/
#'
#' @param x Vector of words, phrases or texts.
#' @param by Vector that indicates how the text should be split.
#' @param model Name of a pretrained model stored on the huggingface.co. (Maybe a path to a  model (.pt or .bin file) stored locally will work.)
#' @ignore_regex Can ignore certain characters when calculates the log probabilities. For example `^[[:punct:]]$` will ignore all punctuation  that stands alone in a token.
#' @eot 0 Does nothing, 1 adds an end of text tag at the beginning of a vector, 2 adds an end of text tag and a space at the beginning of a vector.
#'
#' @return a vector of log probabilities.
#'
#' @examples
#' @export
get_masked_log_prob <- function(x, by = rep(1, length(x)), ignore_regex = "",  model = "distilbert-base-uncased", npred = 0, max_batch_size = 50) {
  get_lm_lp(x =x, by=by, ignore_regex = ignore_regex, type = "masked",model=model,  npred = npred, max_batch_size = max_batch_size)
}


#' Title
#'
#' @param x
#' @param by
#' @param eot
#' @param model
#'
#' @return
#' @export
#'
#' @examples
get_masked_log_prob_mat <- function(x, by = rep(1, length(x)),  model = "distilbert-base-uncased", max_batch_size = 50) {
  x <- trimws(x, whitespace = "[ \t]")
  texts <- split(x, by)
  N <- length(texts)
  tidytable::map2.(texts,names(texts), function(words,item) {
    masked_log_prob_mat(words, model = model,max_batch_size =max_batch_size)

  })
}

masked_log_prob_mat <- function(words, model = "distilbert-base-uncased", max_batch_size = 50){
  text <-  paste0(words, collapse = " ")
  input_ids  <-
    reticulate::py_to_r(tokenizer(model)(text, return_tensors = "pt")$input_ids$tolist())[[1]]
  mask_id <- reticulate::py_to_r(tokenizer(model)$mask_token_id)
  special_ids <- reticulate::py_to_r(tokenizer(model)$all_special_ids)
  # number of masks that need to be applied and where
  nmasks <- sum(!input_ids %in% special_ids)
  masks <- which(!input_ids %in% special_ids) #starts from 1
  n_input_ids <- rep(list(input_ids), nmasks)
  input_ids_masked <- tidytable::map2.(n_input_ids,seq_along(masks), ~ {
    .x[masks[.y:length(masks)]] <- mask_id
    .x
  })
  #makes a tensor for the language model
  tinput_ids_masked <- torch$tensor(input_ids_masked)
  n_tensors <- reticulate::py_to_r(tinput_ids_masked$shape[0])
  n_groups <-  n_tensors/ max_batch_size

  tensor_groups <- split(0:(n_tensors-1), ceiling((1:n_tensors)/max_batch_size))
  tinput_ids_masked_lst <- lapply(tensor_groups, function(i) tinput_ids_masked[i])


  message_verbose("Processing ", n_tensors," tensors in ", length(tinput_ids_masked_lst)," groups of (maximum) ",max(sapply(tensor_groups, length)) , " batches of ",tinput_ids_masked$shape[1]," tokens.")
  message_verbose("Processing using masked model '", model,"'...")

  #out_lm <- lang_model(model, task = "masked")(tinput_ids_masked)
  out_lm_lst <- lapply(tinput_ids_masked_lst, function(t) lang_model(model, task = "masked")(t)$logits)
  out_lm <-  torch$row_stack(unname(out_lm_lst))

  tokens <- reticulate::py_to_r(tokenizer(model)$convert_ids_to_tokens(input_ids[masks]))
  # python objects below, indexes need to start from 0
  # .x is the batch index, iterates over the input_ids_masked
  # .y in the masked word position
  # lp <- tidytable::map2.(0:(nmasks-1),(masks-1), ~ reticulate::py_to_r(torch$log_softmax(out_lm$logits[.x][.y], dim = -1L))$tolist())
  lp <- lapply(1:nmasks, function(n)
  { # n-1 indexes the masked sentence (Starts from 0)
    #masks-1 are the indexes of the masks in the sentences (starts from 0), takes the last n:nmasks masks
    logits <- out_lm[n-1][(masks-1)[n:nmasks]]
    lsm <- torch$log_softmax(logits, dim = -1L)
    mat_mask <- reticulate::py_to_r(lsm)$tolist() |>
      unlist() |>
      matrix(ncol =length(n:nmasks))
    mat_mask <- cbind(matrix(NA, nrow = nrow(mat_mask), ncol = n-1), mat_mask)
    rownames(mat_mask)  <- get_tr_vocab(model)
    colnames(mat_mask) <- unlist(tokens)
    mat_mask
  }
  )
  # stores the predictions from wordn to wordn+1..N in each list
  lp
  # this below doesn't seem to be right
  # lp_by_pred <- lapply(0:(nmasks-1),   function(m){
  #   cbind(matrix(NA, nrow = nrow(lp[[1]]), ncol = m),
  #         lapply(1:(nmasks-m), function(n){
  #           lp[[n]][,n+m,drop = FALSE]
  #         }) |> do.call("cbind",.))
  # })
  #  lp_by_pred
}


# get_masked_log_prior

#https://www.kaggle.com/datasets/toddcook/bert-english-uncased-unigrams
