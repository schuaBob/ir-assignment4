class PseudoRFRetreivalModel:

    indexReader=[]

    def __init__(self, ixReader):
        self.indexReader = ixReader
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


        # get P(token|feedback documents)
        TokenRFScore = self.GetTokenRFScore(query, topK)

        # sort all retrieved documents from most relevant to least, and return TopN
        results=[]
        return results

    def GetTokenRFScore(self, query, topK):
        # for each token in the query, you should calculate token's score in feedback documents: P(token|feedback documents)
        # use Dirichlet smoothing
        # save {token: score} in dictionary TokenRFScore, and return it
        TokenRFScore={}
        return TokenRFScore

