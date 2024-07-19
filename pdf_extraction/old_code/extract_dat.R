#=============================================================================
# Installation of packages for metagear: 
#	install.packages("BiocManager");
#	BiocManager::install("EBImage")
#	devtools::install_github("daniel1noble/metaDigitise")
#=============================================================================
#
#=============================================================================
#For the automated download of PDFs.
#=============================================================================
library(metagear)
#=============================================================================






#=============================================================================
#Old code and notes
#=============================================================================
library(metagear)
library(EBImage)
library(metaDigitise)
library(pdftools)

path1 = "./data/papers/Semchenko2019.pdf"
pdf_convert(path1, format="svg")

# Define the path to the pdfimages executable and PDF file
pdfimages_path = "C:/Users/jusin/AppData/Local/poppler-24.02.0/Library/bin/pdfimages.exe"  # Adjust according to your system
path1 = "./data/papers/Semchenko2019.pdf"
output_prefix = "./data/papers/extracted_image"

# Construct the system call command
command = sprintf('"%s" -all "%s" "%s"', pdfimages_path, path1, output_prefix)

# Execute the command
system(command)


# Define the path to the pdfimages executable and PDF file
pdfimages_path = "C:/Users/jusin/AppData/Local/poppler-24.02.0/Library/bin/pdftocairo.exe"  # Adjust according to your system
path1 = "./data/papers/Semchenko2019.pdf"
output_prefix = "./data/papers/extracted_image"

# Construct the system call command
command = sprintf('"%s" -svg "%s" "%s"', pdfimages_path, path1, output_prefix)

# Execute the command
system(command)



imageFiles = pdf_convert(path1) 
PDF_extractImages("./../../booth1997.pdf")
nfigs = length(imageFiles)

par(mfrow=c(1,nfigs), las = 1)
for(i in 1:nfigs) {
  figure_display(imageFiles[i])
  mtext(imageFiles[i], col = "red", cex = 1.2)
}

im1 = readPNG("semchenko2019_export3.png")                                                                             
writeJPEG(im1, target = "semchenko2019_export3.jpg")                                                                   
rd1 = figure_barPlot("semchenko2019_export3.jpg",axis_sensitivity = 0.5, bar$    