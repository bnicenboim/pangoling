test_that("`.onLoad` initializes correctly", {
 
  # Test global variable initialization
  expect_true(!is.null(pangoling:::transformers))
  expect_true(!is.null(pangoling:::torch))
  
  # Test options are set correctly
  options <- options()
  expect_true("pangoling.debug" %in% names(options))
  expect_true(options$pangoling.causal.default == "gpt2")
  expect_true(options$pangoling.masked.default == "bert-base-uncased")
  
  # Test memoization
  expect_true(memoise::is.memoised(pangoling:::tokenizer))
  expect_true(memoise::is.memoised(pangoling:::lang_model))
  expect_true(memoise::is.memoised(pangoling:::transformer_vocab))
})


test_that("`.onAttach` shows correct startup message", {
  output <- capture.output(pangoling:::.onAttach(libname = "testLib", 
                                                 pkgname = "pangoling"),
                           type = "message")
    # Check that the output contains the correct version
  expect_true(grepl("pangoling version", output[[1]]))
})

test_that("Package options are correctly set and can be overridden", {

  # Check default options
  expect_equal(getOption("pangoling.debug"), FALSE)
  # Override and test new options
  options(pangoling.verbose = 1)
  expect_equal(getOption("pangoling.verbose"), 1)
})



