
options(tibble.width = Inf)
region_order <- c(
  "N", "V", "Det+CL", "Adv", "VN", "FreqP",
  "DE", "head", "hd1", "hd2", "hd3", "hd4", "hd5"
)

exp1 <- tidytable::fread("data-raw/JaegerChenLiLinVasishth2015_Exp1.txt") |>
  tidytable::mutate(region.id = factor(region.id, levels = region_order)) %>%
  tidytable::arrange(subject, item, cond, region.id) |>
  tidytable::mutate(wordn = seq_len(n()), .by= c(item, cond, subject))

exp1 |> print(n=20)

items <- tidytable::fread("data-raw/itemsJaegeretal2014.txt") |>
  tidytable::mutate(sentence = Sentence,
                    Sentence =
                      ifelse(Condition %in% c("a","b"),
                             gsub("^((\\*[^*]*?){3})\\*", "\\1", Sentence),
                             gsub("^((\\*[^*]*?){5})\\*", "\\1", Sentence)
                             )) |>
  tidytable::separate_rows(Sentence, sep = "\\*") |>
  tidytable::filter(Sentence != "") |>
  tidytable::mutate(Sentence = trimws(Sentence)) |>
  tidytable::mutate(wordn = seq_len(n()), .by= c(ItemId, Condition))
print(items, n = 20)
merged <- exp1 %>%
  tidytable::left_join(items,
                       by = c("item" = "ItemId", "cond" = "Condition", "wordn"))

# Select and rearrange columns to match the desired output
df_jaeger14<- merged %>%
  tidytable::select(
    subject, item, cond,
    word = Sentence, wordn,
    rt,
    region = region.id,
    question = Question,
    accuracy,
    correct_answer = Correct_Answer,
    question_type = Question_type,
    experiment = Experiment,
    list = List,
    sentence
  )

Encoding(df_jaeger14$question) <- "UTF-8"
Encoding(df_jaeger14$sentence) <- "UTF-8"

usethis::use_data(df_jaeger14, overwrite = TRUE)

## df_jaeger14 |> print(n=20)
