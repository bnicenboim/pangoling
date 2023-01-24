
# pangoling

<!-- badges: start -->
<!-- badges: end -->

The goal of pangoling is to use transformer models to get
log-probabilities of words. It’s a wrapper of the python package
[`transformer`](https://pypi.org/project/transformers/).

## Installation

There is still no released version of `pangoling`. The package is in the
early stages of development, and it will be subject to a lot of changes.
To install the latest version from github use:

``` r
# install.packages("remotes") # if needed
remotes::install_github("bnicenboim/pangoling")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(pangoling)
```

``` r
library(tidytable) #fast alternative to dplyr
```

The intended use of this package is the following. Given a (toy) dataset
like this.

``` r
sentences <- c("The apple doesn't fall far from the tree.", "Don't judge a book by its cover.")
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
(df_sent <- df_sent |>
  mutate(lp = get_causal_log_prob(word, .by = sent_n)))
#> Processing 1 batch(es) of 10 tokens.
#> Processing using causal model 'gpt2'...
#> Text id: 1
#> `The apple doesn't fall far from the tree.`
#> Processing 1 batch(es) of 9 tokens.
#> Processing using causal model 'gpt2'...
#> Text id: 2
#> `Don't judge a book by its cover.`
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
