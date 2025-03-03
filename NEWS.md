# pangoling (development version)

# pangoling 0.0.0.9000
* First release!

# pangoling 0.0.0.9001
* Tons of stuff. Fully functional package now.

# pangoling 0.0.0.9002
* minor function names to avoid conflict with other packages

# pangoling 0.0.0.9003
* bug in causal_tokens_lp_tbl fixed

# pangoling 0.0.0.9004
* Causal models accept batches. 

# pangoling 0.0.0.9005
* Strings with no tokens no longer throw errors. 
* Requires correct version of R. 

# pangoling 0.0.0.9006
* `causal_lp` get a `l_contexts` argument.
* Checkpoints work for causal models (not yet for masked models).
* Ropensci badge added.


# pangoling 0.0.0.9007
* `set_cache_folder()` function added.
* Message when the package loads.
* New troubleshooting vignette.

# pangoling 0.0.0.9008
* Fix a bug when  `.by` is unordered

# pangoling 0.0.0.9009
* Deprecated `.by` in favor of `by`.

# pangoling 0.0.0.9010

## New Features:
- Added `checkpoint` parameter to `causal_preload()` and `masked_preload()` to 
allow loading models from checkpoints.
- Introduced `causal_next_tokens_pred_tbl()`, which replaces 
`causal_next_tokens_tbl()` and provides improved predictability calculations.
- Added `causal_words_pred()`, `causal_targets_pred()`, and 
`causal_tokens_pred_lst()` to compute predictability for words, phrases, or 
tokens, replacing `causal_lp()` and `causal_tokens_lp_tbl()`.
- Introduced `masked_tokens_pred_tbl()`, replacing `masked_tokens_tbl()`, for 
retrieving possible tokens and their log probabilities.
- Introduced `masked_targets_pred()`, replacing `masked_lp()`, for calculating 
predictability based on left and right context.
- Introduced `transformer_vocab()` with an optional `decode` parameter to return decoded tokenized words.
- **New dataset `df_jaeger14`**: Self-paced reading data on Chinese relative 
clauses.
- **New dataset `df_sent`**: Example dataset with two word-by-word tokenized 
sentences.
- **New vignette**: Added a worked-out example of a causal model.

## Enhancements:
- Added `sep` argument in `causal_words_pred()` to support languages without 
spaces between words (e.g., Chinese).
- New `log.p` argument across multiple functions to specify how predictability 
is calculated (e.g., log base *e*, log base 2 for bits, or raw probabilities).
- Improved tokenization utilities: `tokenize_lst()` now supports decoded outputs 
via the `decode` parameter.
- Updated `install_py_pangoling()` to enhance Python environment handling.
- Added `perplexity_calc()` for computing perplexity from probabilities.

## Deprecations:
- Deprecated `causal_next_tokens_tbl()`, `causal_lp()`, 
`causal_tokens_lp_tbl()`, and `causal_lp_mats()`. Use 
`causal_next_tokens_pred_tbl()`, `causal_targets_pred()`, 
`causal_words_pred()`, and `causal_pred_mats()` instead.
- Deprecated `masked_tokens_tbl()` and `masked_lp()`. Use 
`masked_tokens_pred_tbl()` and `masked_targets_pred()` instead.


# pangoling 0.0.0.9011

- Added `word_n` argument in `causal_words_pred()` to indicate word order of teh texts.