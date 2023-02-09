library(testthat)
library(pangoling)

skip_if_no_python_stuff <- function() {
  have_transformers <- reticulate::py_module_available("transformers")
  have_torch <- reticulate::py_module_available("torch")
  if (!have_transformers) {
    skip("transformers not available for testing")
  }
  if (!have_torch) {
    skip("transformers not available for torch")
  }
}

test_check("pangoling")
