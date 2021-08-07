"""
# sample: https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.py
# source: https://www.sbert.net/examples/applications/semantic-search/README.html
"""
import os

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from model.result import Result
from model.document import Document

# Encode model
encode_model = 'msmarco-MiniLM-L-6-v3'

# Re-rank model
re_rank_model = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


class Retriever:
    def __init__(self, top_k: int = 100):
        self.bi_encoder = SentenceTransformer(encode_model)
        self.cross_encoder = CrossEncoder(re_rank_model)
        self.top_k = top_k
        self.corpus_embeddings_list = None
        self.paragraphs = []
        self.docs_map = {}

    def encode(self, document: Document):
        corpus_embeddings = self.bi_encoder.encode(document.open(), convert_to_tensor=True, show_progress_bar=True)
        torch.save(corpus_embeddings, document.path_pt)

    def load_document(self, document: Document):
        map_location = torch.device('cpu')
        corpus_embeddings = torch.load(document.path_pt, map_location)
        paragraphs = document.open()
        self.docs_map[document.name] = {"paragraphs": paragraphs, "corpus_embeddings": corpus_embeddings}
        # if torch.cuda.is_available():
        #     self.corpus_embeddings = self.corpus_embeddings.to('cuda')

    def load_documents(self, documents: list[Document]):
        for document in documents:
            try:
                self.load_document(document)
            except:
                print(f"error: failed to load {document.name}")
                continue

    def combine_data(self):
        for document_map in self.docs_map:
            corpus_embeddings = self.docs_map[document_map]["corpus_embeddings"]
            paragraphs = self.docs_map[document_map]["paragraphs"]
            if self.corpus_embeddings_list is None:
                self.corpus_embeddings_list = corpus_embeddings
            else:
                self.corpus_embeddings_list = torch.cat([self.corpus_embeddings_list, corpus_embeddings])
            self.paragraphs.extend(paragraphs)

    def remove(self, document):
        if os.path.isfile(document.path_txt):
            os.remove(document.path_txt)
        if os.path.isfile(document.path_pt):
            os.remove(document.path_pt)
        self.docs_map.pop(document.name, None)
        self.combine_data()

    def search(self, query):
        # Semantic Search #
        # Encode the query using the bi-encoder and find potentially relevant passages
        question_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)

        hits = util.semantic_search(question_embedding, self.corpus_embeddings_list, top_k=self.top_k)
        hits = hits[0]  # Get the hits for the first query

        # Re-Ranking #
        # Now, score all retrieved passages with the cross_encoder
        cross_inp = [[query, self.paragraphs[hit['corpus_id']]] for hit in hits]
        cross_scores = self.cross_encoder.predict(cross_inp)

        # Sort results by the cross-encoder scores
        for idx in range(len(cross_scores)):
            hits[idx]['cross-score'] = cross_scores[idx]

        # Output of top-3 hits from bi-encoder
        # print("\n-------------------------\n")
        # print("Top-3 Bi-Encoder Retrieval hits")
        # hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        # for hit in hits[0:3]:
        #     print("\t{:.3f}\t{}".format(hit['score'], self.document.paragraphs[hit['corpus_id']].replace("\n", " ")))

        # Output of top-3 hits from re-ranker
        results = []

        # print("\n-------------------------\n")
        # print("Top-3 Cross-Encoder Re-ranker hits")
        hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
        # answer = None
        for hit in hits[0:3]:
            print("\t{:.3f}\t{}".format(hit['cross-score'], self.paragraphs[hit['corpus_id']].replace("\n", " ")))
            answer = Result(hit['cross-score'], self.paragraphs[hit['corpus_id']].replace("\n", " "))
            results.append(answer)

        return results
        #
        # return result
