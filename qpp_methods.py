import pyterrier as pt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import math

class QPP:
    def __init__(self):
        if not pt.java.started():
            pt.java.init()
        index_path ='/mnt/indices/msmarco-passage.terrier/'
        index_ref = pt.IndexRef.of(index_path)
        self.index = pt.IndexFactory.of(index_ref)
        self.DOC_NUM = self.index.getCollectionStatistics().getNumberOfDocuments()

    def get_max_idf_query(self, qid, q_df):
        stemmer = pt.TerrierStemmer.porter
    
        query = q_df[q_df.qid==qid]['query'].values[0]
        dfs = []
        for token in query.split(' '):
            t = stemmer.stem(token)
            try:
                df = self.index.getLexicon()[t].getDocumentFrequency()
            except:
                df = 1
            dfs.append(df)
            
        max_idf = np.max(np.log((self.DOC_NUM-np.array(dfs)+0.5) / (np.array(dfs)+0.5)))
        return max_idf

    def nqc(self, res, qid, queries=-1, k=100):
        res_q = res[(res.qid == str(qid))]
        res_q = res_q.sort_values(by=['score'], ascending=False)
        res_q = res_q.iloc[:k]
        scores_100 = np.array(res_q.score.values)
        var_value = np.var(scores_100)

        if(queries==-1):
            try:
                queries = res[res.qid == qid][['qid', 'query']].drop_duplicates()
            except:
                print('please provide the query text(s)')
                return -100000
        max_idf = self.get_max_idf_query(qid, queries)
        nqc = var_value * max_idf
        return nqc

    def a_ratio_prediction(self, res, qid, _index):

        ratios = []
        s1 = 0.1
        s2 = 0.2
    
        s = min(50, len(res[res.qid==qid]))
            
        doc_df = res[(res.qid==qid)&(res['rank']<s)]
        doc_embs = _index.vec_loader()(doc_df)['doc_vec'].values
        emb_cluster = np.vstack(doc_embs)
        sim_mat = cosine_similarity(emb_cluster, dense_output=True)
    
        scores = res[(res.qid==qid)&(res['rank']<s)]['score'].values
        scores = np.expand_dims(scores, axis=1)
        score_m_mat = np.dot(scores, scores.T)
    
        sim_mat = sim_mat@score_m_mat
    
        sim_mat_top = sim_mat[:math.ceil(s1*s), :math.ceil(s1*s)].flatten()
        mean_top = np.mean(sim_mat_top)
    
        sim_mat_tail = sim_mat[int(s2*s):, int(s2*s):].flatten()
        mean_tail = np.mean(sim_mat_tail)
            
        prediction = mean_top/mean_tail
        
        return prediction.item()

    def spatial_prediction(self, res, qid, k, q_encoder, _index):
        ratios = []
        query_emb = q_encoder(res[res.qid==qid][['qid', 'query']].iloc[:1])
        query_emb = query_emb['query_vec'].values[0]
        emb_cluster = np.expand_dims(query_emb, axis=0)
            
        doc_df = res[(res.qid==qid)&(res['rank']<k)]
        doc_embs = _index.vec_loader()(doc_df)['doc_vec'].values
        for e in doc_embs:
                # print(e.shape)
            emb_cluster = np.append(emb_cluster, np.expand_dims(e, axis=0), axis=0)
        max_emb = np.max(emb_cluster, axis=0)
        min_emb = np.min(emb_cluster, axis=0)
        edge_emb = max_emb - min_emb
        prediction = -np.sum(np.log(edge_emb))
        
        return prediction.item()