from IndexingWithWhoosh.MyIndexReader import MyIndexReader
from SearchWithWhoosh.QueryRetreivalModel import QueryRetrievalModel
# from Classes.Query import Query
from collections import Counter
from Classes.Document import Document
from math import prod


class PseudoRFRetreivalModel:
    indexReader = []

    def __init__(self, ixReader: MyIndexReader):
        self.indexReader = ixReader
        self.origin_model = QueryRetrievalModel(self.indexReader)
        self.collection_len = self.indexReader.searcher.reader().field_length(
            "doc_content"
        )
        return

    # Search for the topic with pseudo relevance feedback.
    # The returned results (retrieved documents) should be ranked by the score (from the most relevant to the least).
    # query: The query to be searched for.
    # TopN: The maximum number of returned document
    # TopK: The count of feedback documents
    # alpha: parameter of relevance feedback model
    # return TopN most relevant document, in List structure

    def retrieveQuery(self, query, topN, topK, alpha):
        # this method will return the retrieval result of the given Query, and this result is enhanced with pseudo relevance feedback
        # (1) you should first use the original retrieval model to get TopK documents, which will be regarded as feedback documents
        # (2) implement GetTokenRFScore to get each query token's P(token|feedback model) in feedback documents
        # (3) implement the relevance feedback model for each token: combine the each query token's original retrieval score P(token|document) with its score in feedback documents P(token|feedback model)
        # (4) for each document, use the query likelihood language model to get the whole query's new score, P(Q|document)=P(token_1|document')*P(token_2|document')*...*P(token_n|document')
        self.__feedback_docs = self.origin_model.retrieveQuery(query, topK)
        tokens = set(query.getQueryContent().split())
        # get P(token|feedback documents)
        TokenRFScore = self.GetTokenRFScore(query, topK)
        # sort all retrieved documents from most relevant to least, and return TopN
        results = []
        ranks= []
        for doc in self.__feedback_docs:
            wordProbs = []
            doc_content = self.indexReader.searcher.stored_fields(doc.getDocId())[
                "doc_content"
            ].split(" ")
            doc_token_counts = Counter(doc_content)
            for token in tokens:
                if token == "OR":
                    continue
                wordProbs.append(alpha * doc_token_counts[token]/len(doc_content) + TokenRFScore[token])
            ranks.append((prod(wordProbs), doc.getDocId()))
        ranks.sort(reverse=True)
        results = []
        for score, docId in ranks[:topN]:
            d = Document()
            d.setDocId(docId)
            d.setDocNo(self.indexReader.getDocNo(docId))
            d.setScore(score)
            results.append(d)
        assert len(results) == topN
        return results

    def GetTokenRFScore(self, query, topK):
        # for each token in the query, you should calculate token's score in feedback documents: P(token|feedback documents)
        # use Dirichlet smoothing
        # save {token: score} in dictionary TokenRFScore, and return it
        mu = 2000
        doc_token_counts = Counter()
        total_tokens = 0
        TokenRFScore = {}
        for doc in self.__feedback_docs[:topK]:
            doc_content = self.indexReader.searcher.stored_fields(doc.getDocId())[
                "doc_content"
            ].split(" ")
            total_tokens += len(doc_content)
            doc_token_counts.update(doc_content)
        for token in set(query.getQueryContent().split()):
            prob = (
                doc_token_counts[token] + mu * doc_token_counts[token] / total_tokens
            ) / (len(self.__feedback_docs[:topK]) + mu)
            TokenRFScore[token] = prob
        return TokenRFScore
