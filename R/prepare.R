
download_data <- function(source, force= FALSE){
  fileRDS <- paste0(source$name,".RDS")
  if(!force){
    file_1 <- file.path(rappdirs::user_data_dir("pangolang"), fileRDS)
    file_2 <- file.path(tempdir(), fileRDS)
    which_file <- which.max(c(file.mtime(file_1), file.mtime(file_2)))
    if(length(which_file)!=0){
      # subtlexus <<- readRDS(c(file_1, file_2)[which_file])
      assign(source$name, readRDS(c(file_1, file_2)[which_file]), envir=.pkg_env)
      return(invisible())
    }
  }
  file <- file.path(tempdir(),paste0(source$name,".",source$filetype))
  if(interactive()){
    choices <- c("yes","no")
    choice <- menu(choices, title = paste0("Do you want to download ", source$name," data set from [",source$url,"] ?"))
    if(choices[choice] != "yes") stop("Cannot continue without downloading the dataset."
    )
  }
  httr::GET(source$url,
            httr::write_disk(file, overwrite = TRUE),
            httr::progress())
  if(source$filetype =="zip"){
    metadata <- unzip(file, exdir = tempdir(), list = TRUE)
    utils::unzip(file, exdir = tempdir())
    file <- file.path(tempdir(),metadata$Name)
      } else if(source$filetype =="xlsx"){

    #TODO
  }
  data_downloaded <- tidytable::fread.(file)
  attributes(data_downloaded)$download_date <- Sys.Date()
  data_downloaded
}
