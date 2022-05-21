
corpus_sources_lst <- list(Provo_cloze = list(
  description = "The Provo Corpus is a corpus of eye-tracking data with accompanying predictability norms. The predictability norms for the Provo Corpus differ from those of other corpora. In addition to traditional cloze scores that estimate the predictability of the full orthographic form of each word, the Provo Corpus also includes measures of the predictability of morpho-syntactic and semantic information for each word. This makes the Provo Corpus ideal for studying predictive processes in reading. Some analyses using these data have previously been reported elsewhere [Luke, S. G., and Christianson, K. (2016). Limits on lexical prediction during reading. Cognitive Psychology, 88, 22-60.]. Details about the content of the corpus can be found in our paper in Behavior Research Methods [Luke, S.G. and Christianson, K. (Submitted) The Provo Corpus: A Large Eye-Tracking Corpus with Predictability Norms].",
  url = "https://osf.io/e4a2m//?action=download",
  name = "Provo_cloze",
  language = "english",
  location = "US",
  locale = "en_US",
  filetype = "csv",
  doi = c("10.3758/s13428-017-0908-4", "10.1016/j.cogpsych.2016.06.002"))
)
if(0){
# the corpus: https://osf.io/e4a2m/
data_provo_predictability() <- function(){
  provo_tbl <- download_data(corpus_sources_lst$Provo_cloze)

  provo_tbl <- provo_tbl %>% tidytable::rename_with.(tolower)

  provo_h_tbl <- provo_tbl%>%
    tidytable::distinct.(text_id, text) %>%
      tidytable::mutate.(word = chr_split(text, " ")) %>%
    tidytable::unnest_longer.(col = word) %>%
    tidytable::mutate.(word = chr_replace_all(word, '""','"')) %>%
    tidytable::filter.(word !="") %>%
    tidytable::mutate.(word_number = 1:n(), .by = "text_id" )

  h_cloze <- provo_tbl%>%
     tidytable::filter.(word==response | (word!=response & response == response[1]), .by =word_unique_id  ) %>%
    tidytable::distinct.(word_unique_id, text_id, word_number, word, response, response_proportion,
                         response_count, total_response_count) %>%
    tidytable::mutate.()

  provo_cp_tbl <- tidytable::left_join.(provo_h_tbl, h_cloze, by = c("text_id","word_number")) %>%
    tidytable::filter.(text_id !=13) #seems problematic
  provo_CP_tbl <- provo_cp_tbl  %>% tidytable::distinct.(text_id, text, word, word_number, CP= response_proportion) %>%
    tidytable::mutate.(CP = ifelse(is.na(CP),0, CP))
  readr::write_csv(provo_CP_tbl,"provo_CP.csv")

  logp <- pangolang::get_word_log_prob(provo_CP_tbl$word, provo_CP_tbl$text_id)

  provo_cp_tbl %>% filter.(stringr::str_remove(tolower(word),"[:punct:]") != stringr::str_remove(tolower(response),"[:punct:]")) %>%
    select.(word_number, word, response,text_id) %>% print(n =200) %>%
  provo_h_tbl %>% filter(text_id == 13, word_number > 17) %>% print(n=100)
  provo_tbl %>% filter(text_id == 13, word_number > 17) %>%  select.(word_number, word, response) %>% print(n=100)
  provo_tbl %>% filter(text_id == 13, word_number > 17) %>% pull(text)  %>% unique()
  }
}
