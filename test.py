import pyterrier as pt
import pyterrier_rag
import pandas as pd
import argparse
import json
import torch
import numpy as np
import pathlib
from tqdm import tqdm
from qpp_methods import *
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


class ContextSelector:
    def __init__(self, strategy="full", encoder_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.strategy = strategy
        self.encoder_model = encoder_model
        self.tokenizer = None
        self.model = None

        if strategy in ["similarity", 'diversity', 'hybrid']:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
            self.model = AutoModel.from_pretrained(encoder_model)
    
    def encode_text(self, text):
        if isinstance(text,'str'):
            text = [text]
        
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy()


    def select_context(self, original_query, intermediate_queries, context_metadata=None):
        if self.strategy == "full":
            return list(range(len(intermediate_queries)))
        elif self.strategy == "none":
            return []
        elif self.strategy == "first_only":
            return [0] if intermediate_queries else []
        elif self.strategy == "last_only":
            return [len(intermediate_queries) - 1] if intermediate_queries else []
        elif self.strategy == "similarity":
            return self._similarity(original_query, intermediate_queries)
        elif self.strategy == "diversity":
            return self._diversity(intermediate_queries)
        elif self.strategy == "hybrid":
            return self._hybrid(original_query=original_query, intermediate_queries=intermediate_queries)
        elif self.strategy.startswith("fixed_"): # Fixed_1,3 
            indices = [int(x) for x in self.strategy.split('_')[1].split(',')]
            return [i for i in indices if i < len(intermediate_queries)]
        else:
            raise ValueError("Unknown context")


    def _similarity(self, original_query, intermediate_queries, top_k = 3):
        if not intermediate_queries:
            return None
        
        original_embeddings = self.encode_text(original_query)

        intermediate_queries = [ctx[0] for ctx in intermediate_queries]
        intermediate_embeddings = self.encode_text(intermediate_queries)

        similarities = cosine_similarity(original_embeddings, intermediate_embeddings)[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return top_indices.tolist()
    
    def _diversity(self, intermediate_queries, max_contexts=3):
        if not intermediate_queries:
            return []
        
        if len(intermediate_queries) <= max_contexts:
            return list(range(len(intermediate_queries)))
        
        intermediate_queries = [ctx[0] for ctx in intermediate_queries]
        embeddings = self.encode_text(intermediate_queries)

        selected = [0]

        # do greedy algo
        for _ in range(max_contexts-1):
            if len(selected) >= len(intermediate_queries):
                break

            best_match = -1
            best_min_cand = -1

            for i in range(len(intermediate_queries)):
                if i in selected:
                    continue

                min_sim = min([cosine_similarity(embeddings[i:i+1], embeddings[j:j+1])[0][0] for j in selected])

                if min_sim > best_min_cand:
                    best_min_cand = min_sim
                    best_match = i

            if best_match != -1:
                selected.append(best_match)
        
        return selected
    
    def _hybrid(self, original_query, intermediate_queries):
        if not intermediate_queries:
            return []
        
        sim_queries = self._similarity(original_query=original_query, intermediate_queries=intermediate_queries, top_k=5)

        if len(sim_queries) > 3:
            sim_contexts = [(intermediate_queries[i][0], intermediate_queries[i][1]) for i in sim_queries]
            div_contexts = self._diversity(sim_contexts, max_contexts=3)
            return [sim_queries[i] for i in div_contexts]

        return sim_queries

class CustomPipeline:
    def __init__(self, retriever, top_k = 3, model="r1"):
        self.retriever = retriever
        self.top_k = top_k
        self.model = model
        self.context_logs = []

        self.pipelines = self._create_pipelines()

    def _create_pipelines(self):
        pipelines = {}
        configs = ["full_context", "no_context", "original_only"]

        if self.model == "r1":
            for config in configs:
                pipelines[config] = pyterrier_rag.SearchR1(self.retriever, self.top_k)
        elif self.model == "r1s":
            model_kwargs = {
                'tensor_parallel_size': 1,
                'dtype': 'bfloat16',
                'quantization': 'bitsandsbytes',
                'gpu_memory_utilization': 0.7,
                'max_model_len': 92000
            }
            for config in configs:
                pipelines[config] = pyterrier_rag.R1Searcher(
                self.retriever, top_k=self.top_k, verbose=False,
                model_kw_args=model_kwargs)

        
        return pipelines
    
    def run_pipeline(self, queries, answers, config="full_context"):
        pipeline = self.pipelines[config]
        results = []

        batch_size = 10
        for i in tqdm(range(0, len(queries), batch_size), desc=f"Processing {config}"):
            batch_queries = queries.iloc[i:i+batch_size]

            for _, query_row in batch_queries.iterrows():
                qid = query_row['qid']
                query = query_row['query']

                try:
                    result = pipeline.search(query)
                    answer = result.iloc[0]['qanswer'] if not result.empty else None

                    self._log_analysis(qid, query, config, answer)

                    results.append({
                        "qid": qid,
                        "query": query,
                        "qanswer": answer,
                        "config": config
                    })
                
                except Exception as e:
                    print(f"Error processing query {qid}: {e}")
                    results.append({
                        "qid": qid,
                        "query": query,
                        "qanswer": None,
                        "config": config
                    })

        return pd.DataFrame(results)
        
    def _log_analysis(self, qid, query, config, answer):
        log_entry  = {
            'qid': qid,
            'query': query,
            'config': config,
            'answer': answer is not None,
            'answer_length': len(answer) if answer else 0
        }
        self.context_logs.append(log_entry)

    def _save_logs(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.context_logs, f, indent=2)

def run_baseline(retriever, queries, answers, configs, k=3, task="nq_test", model="r1"):
    results = {}

    analyzer = CustomPipeline(retriever=retriever, top_k=k, model=model)

    for config in configs:
        print(f"Running experiment on {config}")

        config_results = analyzer.run_pipeline(queries=queries, answers=answers, config=config)
        eval_results = evaluator(config_results, answers)

        results[config] = {
            "results": config_results,
            "evaluation": eval_results,
            "mean_em": eval_results['em'].mean(),
            "mean_f1": eval_results['f1'].mean(),
        }

        output_filename = f"baseline_{config}_{k}_{task}_{model}.res"
        config_results.to_csv(f"./baseline_res/{output_filename}", index=False)
        eval_results.to_csv(f"./baseline_eval/{output_filename}", index=False)

        print(f"Config: {config}: EM={results[config]['mean_em']:.3f}, F1={results[config]['mean_f1']:.3f}")
    
    log_filename = f"baseline_analysis_{k}_{task}_{model}_logs.json"
    analyzer._save_logs(f"./baseline_logs/{log_filename}")

    return results

def evaluator(res, _ans):
    df_content = []
    
    for qid in res.qid.unique():
        golden_answers = _ans[_ans.qid == qid].gold_answer.to_list()
        
        isnull_indicator = res[res.qid==qid].qanswer.isnull().values[0]
        if(isnull_indicator):
            df_content.append([qid, 0.0, 0.0])
            continue
            
        prediction = res[res.qid==qid].qanswer.values[0]
        em_score = pyterrier_rag._measures.ems(prediction, golden_answers)
        f1_list = []
        for a in golden_answers:
            f1_list.append(pyterrier_rag._measures.f1_score(prediction, a))
        f1_score = max(f1_list)
        
        df_content.append([qid, em_score, f1_score])
    
    return pd.DataFrame(df_content, columns=['qid', 'em', 'f1'])

def log_fn(res, _ret, _ans, _k=3, _task='nq_test', _model='r1'):
    if(_task=='nq_test'):
        output_filename = f"{_ret}_{_k}_{_model}.res"
    else:
        output_filename = f"{_ret}_{_k}_{_task}_{_model}.res"
    csvfile = pathlib.Path(f"./res/{output_filename}")
    eval_csvfile = pathlib.Path(f"./eval_res/{output_filename}")
    res.to_csv(f"./res/{output_filename}", mode='a', index=False, header=not csvfile.exists())
    eval_res = evaluator(res, _ans)
    eval_res.to_csv(f"./eval_res/{output_filename}", mode='a', index=False, header=not eval_csvfile.exists())
    return res

def log_qpp(res, _ret, _k, _index=-1, q_encoder=-1, _task='nq_test', _model='r1'):
    if(_task=='nq_test'):
        output_filename = f"{_ret}_{_k}_{_model}.res"
    else:
        output_filename = f"{_ret}_{_k}_{_task}_{_model}.res"
    csvfile = pathlib.Path(f"./qpp_res/{output_filename}")
    retrieval_res_csvfile = pathlib.Path(f"./retrieval_res/{output_filename}")
    qpp_df_values = []
    for qid in res.qid.unique():
        sorted_res = res[res.qid==qid].sort_values(by=['score'], ascending=False)
        sorted_res.iloc[:200].to_csv(f"./retrieval_res/{output_filename}", mode='a', index=False, header=not retrieval_res_csvfile.exists())
        query_text = res[res.qid==qid]['query'].values[0]
        nqc_est = qpp.nqc(res[res.qid==qid], qid, k=100)
        qpp_df_values.append([qid, query_text, 'nqc', nqc_est, str({'k': 100})])
        if(_index != -1):
            a_ratio_est = qpp.a_ratio_prediction(res[res.qid==qid], qid, _index)
            qpp_df_values.append([qid, query_text, 'a_ratio', a_ratio_est, str({'k': 50, 's1': 0.1, 's2': 0.2})])
            spatial_est = qpp.spatial_prediction(res[res.qid==qid], qid, max(3, _k), q_encoder, _index)
            qpp_df_values.append([qid, query_text, 'spatial', spatial_est, str({'k': max(3, _k)})])
    qpp_df = pd.DataFrame(qpp_df_values, columns=['qid', 'query', 'qpp_method', 'qpp_estimation', 'qpp_parameters']) 
    qpp_df.to_csv(f"./qpp_res/{output_filename}", mode='a', index=False, header=not csvfile.exists())
    return res

def load_retriever(_ret, _task='nq_test', _model='r1'):
    if(not pt.java.started()):
        pt.java.init()
    print(f'Loading retriever {_ret} for {_task}....')
    
    if(_task=='nq_test'):
        artifact = pt.Artifact.from_hf('pyterrier/ragwiki-terrier')
        sparse_index = pt.Artifact.from_hf('pyterrier/ragwiki-terrier')
        bm25 = sparse_index.bm25(include_fields=['docno', 'text', 'title'], threads=5)
        bm25_pipeline = pt.rewrite.tokenise() >> bm25 >> pt.rewrite.reset()
    elif(_task=='hotpotqa_dev'):
        sparse_index_path = '../get_res/hotpotqa_sparse_index'
        index_ref = pt.IndexRef.of(sparse_index_path)
        sparse_index = pt.IndexFactory.of(index_ref)
        bm25 = pt.terrier.Retriever(sparse_index, wmodel='BM25')
        bm25_pipeline = pt.rewrite.tokenise() >> bm25 >> sparse_index.text_loader(["text", "title"]) >> pt.rewrite.reset() 

    if(_ret == 'bm25'):
        return bm25_pipeline >> pt.apply.generic(lambda x : log_qpp(x, _ret, k, _task=_task, _model=_model))
    elif(_ret == 'monoT5'):
        from pyterrier_t5 import MonoT5ReRanker
        monoT5 = MonoT5ReRanker(batch_size=64, verbose=False)
        return (bm25_pipeline % 20) >> monoT5 >> pt.apply.generic(lambda x : log_qpp(x, _ret, k, _task=_task, _model=_model))
    elif(_ret =='E5'):
        from pyterrier_dr import E5
        import pyterrier_dr
        
        if(_task=='nq_test'):
            e5_index = pt.Artifact.from_hf('pyterrier/ragwiki-e5.flex')
        elif(_task=='hotpotqa_dev'):
            e5_index = pyterrier_dr.FlexIndex('../get_res/e5_hotpotqa_wiki_index_2.flex')

        e5_query_encoder = E5()
        e5_ret = e5_query_encoder >> e5_index.torch_retriever(fp16=True, num_results=120) >> sparse_index.text_loader(["text", "title"])
        return e5_ret >> pt.apply.generic(lambda x : log_qpp(x, _ret, k, e5_index, e5_query_encoder, _task=_task, _model=_model))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", type=str, default='bm25', choices=['bm25', 'monoT5', 'E5'])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--model", type=str, default='r1', choices=['r1', 'r1s'])
    parser.add_argument("--task", type=str, default='nq_test', choices=['nq_test', 'hotpotqa_dev'])
    parser.add_argument("--experiment", type=str, default="original", choices=['original', 'baseline'])
    parser.add_argument("--configs", type=str, nargs='+', default=['full_context', 'no_context','original_only'])
    # NEW ARGUMENT FOR DATASET SIZE
    parser.add_argument("--subset_size", type=int, default=None, help="Number of samples to use from the dataset (default: use full dataset)")
    
    args = parser.parse_args()
    k = args.k
    model = args.model
    ret = args.retriever
    task = args.task
    experiment = args.experiment
    configs = args.configs
    subset_size = args.subset_size

    pathlib.Path("./baseline_res").mkdir(exist_ok=True)
    pathlib.Path("./baseline_eval").mkdir(exist_ok=True)
    pathlib.Path("./baseline_logs").mkdir(exist_ok=True)
    
    # Load datasets
    if(task=='nq_test'):
        queries = pt.get_dataset('rag:nq').get_topics('test')
        answers = pt.get_dataset('rag:nq').get_answers('test')
    elif(task=='hotpotqa_dev'):
        queries = pd.read_csv('./hotpotqa_materials/hotpotqa_queries.csv')
        answers = pd.read_csv('./hotpotqa_materials/hotpotqa_answers.csv')

    # SUBSET THE DATA IF REQUESTED
    if subset_size is not None:
        print(f"Using subset of {subset_size} samples from the dataset")
        queries = queries.head(subset_size)
        # Filter answers to match the subset of queries
        subset_qids = queries['qid'].tolist()
        answers = answers[answers['qid'].isin(subset_qids)]
        print(f"Subset created: {len(queries)} queries, {len(answers)} answers")
    else:
        print(f"Using full dataset: {len(queries)} queries, {len(answers)} answers")

    retriever = load_retriever(ret, task, model)

    if experiment == "original":
        print("Running the original pipeline...")
        qpp = QPP()
        if(model=='r1'):
            print('Loading R1 pipeline ....')
            r1_pipeline = pyterrier_rag.SearchR1(retriever, retrieval_top_k=k) >> pt.apply.generic(lambda x : log_fn(x, ret, answers, k, task, model))
            print('R1 pipeline is now loaded!')
        elif(model=='r1s'):
            print('Loading R1-Searcher pipeline ....')
            r1_pipeline = pyterrier_rag.R1Searcher(retriever, top_k=k, verbose=False, model_kw_args={'tensor_parallel_size':1, 'dtype':'bfloat16', 'quantization':"bitsandbytes", 'gpu_memory_utilization':0.6, 'max_model_len':92000}) >> pt.apply.generic(lambda x : log_fn(x, ret, answers, k, task, model))
            print('R1-Searcher pipeline is now loaded!')

        if(task=='nq_test'):
            output_filename = f"{ret}_{k}_{model}.res"
        else:
            output_filename = f"{ret}_{k}_{task}_{model}.res"

        # MODIFIED: Check existing results based on subset size
        try:
            existing_res_df = pd.read_csv(f"./res/{output_filename}")
            num_existing_qids = existing_res_df.shape[0]
            if subset_size is not None:
                # If using subset, start from 0 to avoid confusion
                num_existing_qids = 0
                print("Starting fresh due to subset usage")
        except:
            num_existing_qids = 0

        print(f'Retrieval pipeline {ret} for {task} is now loaded!')

        _batch_size = 10

        print(f'Start generation! k={k}, processing {len(queries)} queries')
        for i in tqdm(range(num_existing_qids, len(queries), _batch_size)):
            _batch_queries = queries.iloc[i: i+_batch_size]
            r1_pipeline(_batch_queries)
    
    elif experiment == "baseline":
        print("Running the baseline experiment...")
        results = run_baseline(retriever, queries, answers, configs, k=k, task=task, model=model)