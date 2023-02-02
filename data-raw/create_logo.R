imgurl <- "inst/figures/pangolin_abs_dark.jpeg"
filename="inst/figures/hex-pangoling.png"
hexSticker::sticker(imgurl, package="pangoling", p_size=20, s_x=1, s_y=1, s_width=1, p_color= "white",
        filename=filename,
        white_around_sticker = TRUE)

fuzz <- 50
library(magick)
#https://stackoverflow.com/questions/60426922/trim-around-hexagon-shape-with-hexsticker
p <- image_read(filename)
pp <- p %>%
  image_fill(color = "transparent", refcolor = "white", fuzz = fuzz, point = "+1+1") %>%
  image_fill(color = "transparent", refcolor = "white", fuzz = fuzz, point = paste0("+", image_info(p)$width-1, "+1")) %>%
  image_fill(color = "transparent", refcolor = "white", fuzz = fuzz, point = paste0("+1", "+", image_info(p)$height-1)) %>%
  image_fill(color = "transparent", refcolor = "white", fuzz = fuzz, point = paste0("+", image_info(p)$width-1, "+", image_info(p)$height-1))
image_write(image = pp, path = filename)
image_write(image = pp, path = filename)

file.show(filename)
