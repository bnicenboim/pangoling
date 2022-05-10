model = "gpt2"
device = "cpu"
text <- "the cat is crazy"
microbenchmark::microbenchmark(
  for(n in 1:20){
    LM <- incremental_LM_scorer(model, device)
    py_logprobs <- LM$logprobs(LM$prepare_text(text))},
  { LM <- incremental_LM_scorer(model, device)
    for(n in 1:20){
    py_logprobs <- LM$logprobs(LM$prepare_text(text))}
    },
  times = 10
)
