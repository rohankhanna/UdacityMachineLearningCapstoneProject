var markdownpdf = require("markdown-pdf")
  , fs = require("fs")

fs.createReadStream("./report.md")
  .pipe(markdownpdf())
  .pipe(fs.createWriteStream("./report.pdf"))

// --- OR ---

//markdownpdf().from("/path/to/document.md").to("/path/to/document.pdf", function () {
//  console.log("Done")
//})
