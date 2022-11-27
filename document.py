class Document(object):
    def __init__(self, docno=None, month=None, date=None, year=None, docid=None, headline=None, rawFilePath=None) -> None:
        self.docno = docno
        self.month = month
        self.date = date
        self.year = year
        self.docid = docid
        self.headline = headline
        self.rawFilePath = rawFilePath
        
    