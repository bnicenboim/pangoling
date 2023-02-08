options(pangoling.verbose = FALSE)

test_that("bert masked works", {
  skip_if_no_python_stuff()
  mask_1 <-
    get_masked_tokens_tbl("The apple doesn't fall far from the [MASK].")
  expect_snapshot(mask_1)
  mask_2 <-
    get_masked_tokens_tbl("The apple doesn't fall far from [MASK] [MASK].")
  expect_snapshot(mask_2)
  mask_2_ <-
    get_masked_tokens_tbl("[CLS] The apple doesn't fall far from [MASK] [MASK]. [SEP]", add_special_tokens = FALSE)
  expect_equal(mask_2, mask_2_)
})

get_masked_tokens_tbl("The apple doesn't fall far from the [MASK].", )
