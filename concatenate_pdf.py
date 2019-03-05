from PyPDF2 import PdfFileMerger

pdfs = ['C:\\Users\\xianz.REDMOND\\Documents\\personal\\visa_1.pdf', 'C:\\Users\\xianz.REDMOND\\Documents\\personal\\visa_2.pdf']

merger = PdfFileMerger()

for pdf in pdfs:
    merger.append(open(pdf, 'rb'))

with open('result.pdf', 'wb') as fout:
    merger.write(fout)