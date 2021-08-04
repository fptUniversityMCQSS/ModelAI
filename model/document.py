import re
import os
from os import walk
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import io

folder_txt = "data/txt"
folder_pt = "data/pt"


class Document:
    def __init__(self, filename):
        self.name = os.path.splitext(filename)[0]
        self.paragraphs = None
        self.path_txt = folder_txt + "/" + self.name + '.txt'
        self.path_pt = folder_pt + "/" + self.name + '.pt'
        self.status = False

    def open(self):
        with open(self.path_txt, encoding="utf8") as txtFile:
            self.paragraphs = [line.rstrip() for line in txtFile]


def list_documents():
    os.makedirs(folder_txt, exist_ok=True)
    os.makedirs(folder_pt, exist_ok=True)
    documents = []
    for (dir_path, dir_names, filenames) in walk(folder_txt):
        for filename in filenames:
            document = Document(filename)
            document.open()
            documents.append(document)
            break
        break
    return documents


def from_pdf_to_document(pdf_stream, filename):
    rs_manager = PDFResourceManager()
    string_io = io.StringIO()
    # codec = 'utf-8'
    la_params = LAParams()
    device = TextConverter(rs_manager, string_io, laparams=la_params)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rs_manager, device)
    # Process each page contained in the document.

    for page in PDFPage.get_pages(pdf_stream):
        interpreter.process_page(page)
        data = string_io.getvalue()

    filename = os.path.splitext(filename)[0]
    document = Document(filename)
    document.paragraphs = to_paragraphs(data)

    with open(document.path_txt, mode="w", encoding="utf8") as txt_file:
        for line in document.paragraphs:
            txt_file.write(line + '\n')

    return document


def to_paragraphs(data):
    paragraphs = []
    new_line = ""
    for line in data.splitlines():
        if len(line) > 1:
            if line[-1] == '.':
                new_line += line
                paragraphs.append(re.sub(r"\s+", " ", new_line))
                new_line = ""
            elif line[-1] == '-':
                new_line += line[:-1]
            else:
                new_line += line + ' '
    return paragraphs
