
# pangoling <a href="http://bruno.nicenboim.me/pangoling/"><img src="man/figures/logo.png" align="right" height="139" /></a>

<!-- badges: start -->

[![Codecov test
coverage](https://codecov.io/gh/bnicenboim/pangoling/branch/main/graph/badge.svg)](https://app.codecov.io/gh/bnicenboim/pangoling?branch=main)
[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![R-CMD-check](https://github.com/bnicenboim/pangoling/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/bnicenboim/pangoling/actions/workflows/R-CMD-check.yaml)
[![Project Status: WIP – Initial development is in progress, but there
has not yet been a stable, usable release suitable for the
public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
<!-- badges: end -->

`pangoling`[^1] is an R package for estimating the log-probabilities of
words in a given context using transformer models. The package provides
an interface for utilizing pre-trained transformer models (such as
~~BERT or~~ GPT-2) to obtain word probabilities. These log-probabilities
are often utilized as predictors in psycholinguistic studies. This
package can be useful for researchers in the field of psycholinguistics
who want to leverage the power of transformer models in their work.

The package is mostly a wrapper of the python package
[`transformers`](https://pypi.org/project/transformers/) to process data
in a convenient format. At the moment only “causal” models such as GPT-2
are working.

## Important! Limitations and bias

The training data of the most popular models (such as GPT-2) haven’t
been released, so one cannot inspect it. It’s clear that the data
contain a lot of unfiltered content from the internet, which is far from
neutral. See for example the scope in the [openAI team’s model card for
GPT-2](https://github.com/openai/gpt-2/blob/master/model_card.md#out-of-scope-use-cases),
but it should be the same for many other models, and the [limitations
and bias section of GPT-2 in hugging face
website](https://huggingface.co/gpt2).

## Installation

There is still no released version of `pangoling`. The package is in the
**very early** stages of development, it’s **not well tested**, and it
will be subject to a lot of changes. To install the latest version from
github use:

``` r
# install.packages("remotes") # if needed
remotes::install_github("bnicenboim/pangoling")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(pangoling)
library(tidytable) #fast alternative to dplyr
```

The intended use of this package is the following. Given a (toy) dataset
like this.

``` r
sentences <- c("The apple doesn't fall far from the tree.", 
               "Don't judge a book by its cover.")
(df_sent <- strsplit(x = sentences, split = " ") |> 
  map_dfr(.f =  ~ data.frame(word = .x), .id = "sent_n"))
#> # A tidytable: 15 × 2
#>    sent_n word   
#>     <int> <chr>  
#>  1      1 The    
#>  2      1 apple  
#>  3      1 doesn't
#>  4      1 fall   
#>  5      1 far    
#>  6      1 from   
#>  7      1 the    
#>  8      1 tree.  
#>  9      2 Don't  
#> 10      2 judge  
#> 11      2 a      
#> 12      2 book   
#> 13      2 by     
#> 14      2 its    
#> 15      2 cover.
```

It’s straight-forward to get the log-probability (`-suprisal`) of each
word based on GPT-2.

``` r
df_sent <- df_sent |>
  mutate(lp = get_causal_log_prob(word, .by = sent_n))
#> Processing 1 batch(es) of 10 tokens.
#> Processing using causal model 'gpt2'...
#> Text id: 1
#> `The apple doesn't fall far from the tree.`
#> Processing 1 batch(es) of 9 tokens.
#> Processing using causal model 'gpt2'...
#> Text id: 2
#> `Don't judge a book by its cover.`
df_sent
#> # A tidytable: 15 × 3
#>    sent_n word         lp
#>     <int> <chr>     <dbl>
#>  1      1 The      NA    
#>  2      1 apple   -10.9  
#>  3      1 doesn't  -5.50 
#>  4      1 fall     -3.60 
#>  5      1 far      -2.91 
#>  6      1 from     -0.745
#>  7      1 the      -0.207
#>  8      1 tree.    -1.58 
#>  9      2 Don't    NA    
#> 10      2 judge    -6.27 
#> 11      2 a        -2.33 
#> 12      2 book     -1.97 
#> 13      2 by       -0.409
#> 14      2 its      -0.257
#> 15      2 cover.   -1.38
```

[^1]: The logo of the package was created with [stable
    diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion)
    and the R package
    [hexSticker](https://github.com/GuangchuangYu/hexSticker).
