import IndexingWithWhoosh.MyIndexReader as MyIndexReader
import SearchWithWhoosh.QueryRetreivalModel as QueryRetreivalModel
import SearchWithWhoosh.ExtractQuery as ExtractQuery
import PseudoRFSearch.PseudoRFRetrievalModel as PseudoRFRetrievalModel
import datetime


# This is for INFSCI 2140 in Fall 2023

if __name__ == '__main__':
    startTime = datetime.datetime.now()
    index = MyIndexReader.MyIndexReader("trectext")
    pesudo_search = PseudoRFRetrievalModel.PseudoRFRetreivalModel(index)
    extractor = ExtractQuery.ExtractQuery()
    queries= extractor.getQuries()
    for query in queries:
        print(query.topicId,"\t",query.queryContent)
        results = pesudo_search.retrieveQuery(query, 20, 100, 0.4)
        rank = 1
        for result in results:
            print(query.getTopicId()," Q0 ",result.getDocNo(),' ',rank," ",result.getScore()," MYRUN",)
            rank += 1

    endTime = datetime.datetime.now()
    print("query search time: ", endTime - startTime)
