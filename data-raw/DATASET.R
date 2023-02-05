## code to prepare `DATASET` dataset goes here
library(tidytable)

provo_et <- list(
  url = "https://osf.io/a32be///?action=download",
  name = "provo_et",
  filetype = "csv"
)
provo_cloze <- list(
  description = "The Provo Corpus is a corpus of eye-tracking data with accompanying predictability norms. The predictability norms for the Provo Corpus differ from those of other corpora. In addition to traditional cloze scores that estimate the predictability of the full orthographic form of each word, the Provo Corpus also includes measures of the predictability of morpho-syntactic and semantic information for each word. This makes the Provo Corpus ideal for studying predictive processes in reading. Some analyses using these data have previously been reported elsewhere [Luke, S. G., and Christianson, K. (2016). Limits on lexical prediction during reading. Cognitive Psychology, 88, 22-60.]. Details about the content of the corpus can be found in our paper in Behavior Research Methods [Luke, S.G. and Christianson, K. (Submitted) The Provo Corpus: A Large Eye-Tracking Corpus with Predictability Norms].",
  url = "https://osf.io/e4a2m//?action=download",
  name = "Provo_cloze",
  language = "english",
  location = "US",
  locale = "en_US",
  filetype = "csv",
  doi = c("10.3758/s13428-017-0908-4", "10.1016/j.cogpsych.2016.06.002")
)


provo_tbl <- download_data(provo_et)
# description:
# https://link.springer.com/article/10.3758/s13428-017-0908-4/tables/2
provo_clz <- provo_tbl %>%
  tidytable::rename_with.(tolower) %>%
  tidytable::distinct.(word_unique_id, text_id, word_number, word, word_cleaned, word_length, total_response_count,
    cp = orthographicmatch, modal_response = modalresponse,
    pos_claws, word_content_or_function, word_pos, pos_cp = posmatch,
    lsa_context_score, lsa_response_match_score
  ) %>%
  tidytable::filter.(!is.na(word))

bad <- provo_clz %>%
  tidytable::summarize.(unique(diff(word_number)), .by = "text_id") %>%
  tidytable::filter.(V1 != 1) %>%
  tidytable::pull.(V1)
provo_clz <- provo_clz %>% tidytable::filter.(!text_id %in% bad)

# to extract the first word and the entire sentence.
provo_full_cloze <- download_data(provo_cloze)

provo_first_word <- provo_full_cloze %>%
  tidytable::rename_with.(tolower) %>%
  tidytable::distinct.(text_id, text) %>%
  tidytable::mutate.(word_number = 1, word = strsplit(text, " ")) %>%
  tidytable::mutate_rowwise.(word = word[[1]][[1]]) %>%
  tidytable::mutate.(word = unlist(word)) %>%
  tidytable::filter.(!text_id %in% bad) %>%
  tidytable::mutate.(
    word_cleaned = chr_remove(tolower(word), "[[:punct:]]"),
    word_length = nchar(word_cleaned), word_unique_id = "QID0"
  )

provo_cloze <- tidytable::bind_rows.(provo_clz, provo_first_word %>% tidytable::select.(-text)) %>% tidytable::arrange.(text_id, word_number)

data_provo_cloze <- provo_cloze %>% tidytable::left_join.(provo_first_word %>% tidytable::select.(text_id, text))

data_provo_cloze <- data_provo_cloze %>% rename.(word_n = word_number)

# tidytable::map_chr.(colnames(data_provo_cloze), ~  paste0("#' * ", .x," explain\n")) %>%
#   cat(.,"\n")



url_franks <- "http://stefanfrank.info/readingdata/Data.zip"
# https://doi.org/10.3758/s13428-012-0313-y
file <- paste0(tempfile(), ".zip")
httr::GET(
  url_franks,
  httr::write_disk(file, overwrite = TRUE),
  httr::progress()
)
metadata <- unzip(file, exdir = tempdir(), list = TRUE)
unzip(file, exdir = tempdir())
data_spr <- fread.(file.path(tempdir(), "selfpacedreading.RT.txt")) %>%
  rename.(subj = subj_nr, sent_id = sent_nr, word_n = word_pos) %>%
  mutate.(acc_comprehension = case_when.(
    correct == "c" ~ 1,
    correct == "e" ~ 0,
    TRUE ~ NA
  ), correct = NULL)
data_spr_subj <- fread.(file.path(tempdir(), "selfpacedreading.subj.txt")) %>%
  rename.(subj = subj_nr)

data_spr <- data_spr %>% left_join.(data_spr_subj)

stimuli_pos <- fread.(file.path(tempdir(), "stimuli_pos.txt"), sep = "\t")
names(stimuli_pos) <- c("sent_id", "pos")
stimuli <- fread.(file.path(tempdir(), "stimuli.txt"), sep = "\t")
data_frank2013_stimuli <- stimuli %>%
  rename.(sent_id = sent_nr) %>%
  left_join.(stimuli_pos)

data_frank2013_spr_complete <- data_spr %>%
  mutate.(typo = case_when.(sent_id == 43 & word == "Sott" ~ 1, sent_id == 269 & word == "that" ~ 1, sent_id == 337 & word == "Margeret" ~ 1, TRUE ~ 0)) %>%
  rename.(correct_perc = correct, sent_n = sent_pos)

data_frank2013_spr <- data_frank2013_spr_complete %>%
  filter.(age_en == 0, correct_perc > .8, !sent_id %in% c(43, 269, 337)) %>%
  select.(-age_en, -typo)


data_frank2013_et_rt <- fread.(file.path(tempdir(), "eyetracking.RT.txt"), sep = "\t") %>%
  rename.(subj = subj_nr, acc_comprehension = correct, word_n = word_pos, sent_n = sent_pos, sent_id = sent_nr)

data_frank2013_et_fix <- fread.(file.path(tempdir(), "eyetracking.fix.txt"), sep = "\t") %>%
  rename.(subj = subj_nr, word_n = word_pos, sent_id = sent_nr, letter_n = letter_pos)

# stimuli_pos <- stimuli_pos %>% mutate.(pos = chr_replace_all(pos," \\.",".") %>%
#                           chr_replace_all(" ,",",")) %>%
#                         separate_rows.("pos") %>%
#   mutate.(word_number = 1:n(), .by ="sent_id")

# surprisal values
# Frank, S.L. (2013). Uncertainty reduction as a measure of cognitive load in sentence comprehension. Topics in Cognitive Science.
url_franks2 <- "http://stefanfrank.info/TopiCS2013/data.zip"
file2 <- paste0(tempfile(), "2.zip")
httr::GET(
  url_franks2,
  httr::write_disk(file2, overwrite = TRUE),
  httr::progress()
)
metadata <- unzip(file2, exdir = tempdir(), list = TRUE)
unzip(file2, exdir = tempdir())
data_surp <- fread.(file.path(tempdir(), "info.txt")) %>%
  rename.(sent_id = sent_nr, word_n = word_pos)
data_frank2013_stimuli <- data_frank2013_stimuli %>% left_join.(data_surp)


### Natural stories corpus:

# https://link.springer.com/article/10.1007/s10579-020-09503-7
# https://github.com/languageMIT/naturalstories


url_natural <- "https://raw.githubusercontent.com/languageMIT/naturalstories/master/naturalstories_RTS/"

file_stories <- file.path(tempdir(), "all_stories.tok")
file_batch1 <- file.path(tempdir(), "batch1_pro.csv")
file_batch2 <- file.path(tempdir(), "batch2_pro.csv")
httr::GET(
  paste0(url_natural, "all_stories.tok"),
  httr::write_disk(file_stories, overwrite = TRUE),
  httr::progress()
)
httr::GET(
  paste0(url_natural, "batch1_pro.csv"),
  httr::write_disk(file_batch1, overwrite = TRUE),
  httr::progress()
)
httr::GET(
  paste0(url_natural, "batch2_pro.csv"),
  httr::write_disk(file_batch2, overwrite = TRUE),
  httr::progress()
)
# based on https://github.com/languageMIT/naturalstories/blob/master/naturalstories_RTS/process_RTs.R

b1 <- fread.(file_batch1)
b2 <- fread.(file_batch2)
words <- fread.(file_stories)

b <- bind_rows.(b1, b2)

offset <- 230

data_natural_spr <- b %>%
  filter.(!(item == 3 & zone == offset + 1)) %>%
  mutate.(zone = if_else(item == 3 & zone > offset, zone - 3, zone - 2)) %>%
  inner_join.(words) %>%
  # filter(RT > 100, RT < 3000, correct > 4) %>%
  arrange.(WorkerId, item, zone) %>%
  rename.(subj = WorkerId, word_n = zone) %>%
  select.(-WorkTimeInSeconds) %>%
  mutate.(subj = as.numeric(factor(subj)))


# Item is the story number, zone is the region where the word falls within the story. Note that some wordforms in all_stories.tok differ from those in words.tsv, reflecting typos in the SPR experiment as run.



usethis::use_data(data_provo_cloze, data_frank2013_stimuli, data_frank2013_spr, data_frank2013_spr_complete, data_frank2013_et_fix, data_frank2013_et_rt,
  data_natural_spr,
  overwrite = TRUE
)
