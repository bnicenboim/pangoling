# bert masked works

    Code
      mask_1
    Output
      # A tidytable: 30,522 x 4
         masked_sentence                             token     lp mask_n
         <chr>                                       <chr>  <dbl>  <int>
       1 The apple doesn't fall far from the [MASK]. world  -2.91      1
       2 The apple doesn't fall far from the [MASK]. tree   -4.03      1
       3 The apple doesn't fall far from the [MASK]. sun    -4.12      1
       4 The apple doesn't fall far from the [MASK]. table  -4.13      1
       5 The apple doesn't fall far from the [MASK]. wall   -4.35      1
       6 The apple doesn't fall far from the [MASK]. ground -4.45      1
       7 The apple doesn't fall far from the [MASK]. city   -4.62      1
       8 The apple doesn't fall far from the [MASK]. game   -4.66      1
       9 The apple doesn't fall far from the [MASK]. water  -4.66      1
      10 The apple doesn't fall far from the [MASK]. house  -4.67      1
      # ... with 30,512 more rows

---

    Code
      mask_2
    Output
      # A tidytable: 61,044 x 4
         masked_sentence                                token     lp mask_n
         <chr>                                          <chr>  <dbl>  <int>
       1 The apple doesn't fall far from [MASK] [MASK]. the   -0.964      1
       2 The apple doesn't fall far from [MASK] [MASK]. my    -2.72       1
       3 The apple doesn't fall far from [MASK] [MASK]. a     -2.86       1
       4 The apple doesn't fall far from [MASK] [MASK]. his   -3.29       1
       5 The apple doesn't fall far from [MASK] [MASK]. her   -3.38       1
       6 The apple doesn't fall far from [MASK] [MASK]. it    -3.92       1
       7 The apple doesn't fall far from [MASK] [MASK]. their -3.98       1
       8 The apple doesn't fall far from [MASK] [MASK]. to    -4.21       1
       9 The apple doesn't fall far from [MASK] [MASK]. your  -4.28       1
      10 The apple doesn't fall far from [MASK] [MASK]. this  -4.30       1
      # ... with 61,034 more rows

