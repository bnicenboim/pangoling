## code to prepare `DATASET` dataset goes here

provo_et <- list(url = "https://osf.io/a32be///?action=download",
                 name = "provo_et",
                 filetype = "csv")
provo_cloze = list(
  description = "The Provo Corpus is a corpus of eye-tracking data with accompanying predictability norms. The predictability norms for the Provo Corpus differ from those of other corpora. In addition to traditional cloze scores that estimate the predictability of the full orthographic form of each word, the Provo Corpus also includes measures of the predictability of morpho-syntactic and semantic information for each word. This makes the Provo Corpus ideal for studying predictive processes in reading. Some analyses using these data have previously been reported elsewhere [Luke, S. G., and Christianson, K. (2016). Limits on lexical prediction during reading. Cognitive Psychology, 88, 22-60.]. Details about the content of the corpus can be found in our paper in Behavior Research Methods [Luke, S.G. and Christianson, K. (Submitted) The Provo Corpus: A Large Eye-Tracking Corpus with Predictability Norms].",
  url = "https://osf.io/e4a2m//?action=download",
  name = "Provo_cloze",
  language = "english",
  location = "US",
  locale = "en_US",
  filetype = "csv",
  doi = c("10.3758/s13428-017-0908-4", "10.1016/j.cogpsych.2016.06.002"))

provo_tbl <- download_data(provo_et)
# description:
#https://link.springer.com/article/10.3758/s13428-017-0908-4/tables/2
provo_clz <- provo_tbl  %>% tidytable::rename_with.(tolower) %>%
  tidytable::distinct.(word_unique_id,text_id, word_number,  word, word_cleaned,word_length, total_response_count, cp = orthographicmatch, modal_response = modalresponse,
                       pos_claws, word_content_or_function, word_pos, pos_cp = posmatch,
                       lsa_context_score, lsa_response_match_score) %>%
  tidytable::filter.(!is.na(word))

bad <- provo_clz %>% tidytable::summarize.(unique(diff(word_number)), .by = "text_id") %>% tidytable::filter.(V1 != 1) %>% tidytable::pull.(V1)
provo_clz <- provo_clz %>% tidytable::filter.(!text_id %in% bad)

#to extract the first word and the entire sentence.
provo_full_cloze <- download_data(provo_cloze)

provo_first_word <-  provo_full_cloze %>% tidytable::rename_with.(tolower) %>%
  tidytable::distinct.(text_id, text) %>%
  tidytable::mutate.(word_number = 1,  word = strsplit(text, " ") ) %>%
  tidytable::mutate_rowwise.(word = word[[1]][[1]]) %>%
  tidytable::mutate.(word = unlist(word)) %>%
  tidytable::filter.(!text_id %in% bad) %>%
  tidytable::mutate.(word_cleaned = chr_remove(tolower(word), "[[:punct:]]"),
                     word_length = nchar(word_cleaned), word_unique_id = "QID0")

provo_cloze <- tidytable::bind_rows.(provo_clz, provo_first_word %>% tidytable::select.(-text)) %>% tidytable::arrange.(text_id, word_number)

data_provo_cloze <- provo_cloze %>% tidytable::left_join.(provo_first_word %>% tidytable::select.(text_id, text))

# tidytable::map_chr.(colnames(data_provo_cloze), ~  paste0("#' * ", .x," explain\n")) %>%
#   cat(.,"\n")


usethis::use_data(data_provo_cloze, overwrite = TRUE)

