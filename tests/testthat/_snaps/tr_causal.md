# gpt2 get prob work

    Code
      cont
    Output
      # A tidytable: 50,257 x 2
         token       lp
         <chr>    <dbl>
       1 Ġtree   -0.281
       2 Ġtrees  -3.60 
       3 Ġapple  -4.29 
       4 Ġtable  -4.50 
       5 Ġhead   -4.83 
       6 Ġmark   -4.86 
       7 Ġcake   -4.91 
       8 Ġground -5.08 
       9 Ġtruth  -5.31 
      10 Ġtop    -5.36 
      # ... with 50,247 more rows

---

    Code
      lp_prov
    Output
              The       apple     doesn't        fall         far        from 
               NA -10.9005165  -5.4998987  -3.5977476  -2.9118700  -0.7454544 
              the        tree 
       -0.2066508  -0.2807994 

---

    Code
      lp_prov2
    Output
              The       apple     doesn't        fall         far        from 
               NA -10.9005165  -5.4998987  -3.5977476  -2.9118700  -0.7454544 
              the       tree. 
       -0.2066523  -1.5817280 

# can handle extra parameters

    Code
      probs
    Output
           This        is        it 
      -4.858031 -1.694939 -6.464006 

# other models using get prob work

    Code
      causal_lp(x = c("El", "bebé", "de", "cigüeña."), model = "flax-community/gpt-2-spanish")
    Output
              El       bebé         de   cigüeña. 
              NA  -7.397457  -2.747046 -16.525037 

---

    Code
      lp_provd
    Output
              The       apple     doesn't        fall         far        from 
               NA -14.9577360  -7.0869579  -5.0421410  -4.6047726  -2.2611685 
              the       tree. 
       -0.8102559  -4.3868304 

