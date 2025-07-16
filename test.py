import pyterrier as pt
import pyterrier_rag
import pandas as pd
import argparse
import json
import torch
import numpy as np
import pathlib
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import time


class QuickTestPipeline:
    def __init__(self, retriever, top_k=3, model="r1", sample_size=20):
        self.retriever = retriever
        self.top_k = top_k
        self.model = model
        self.sample_size = sample_size
        self.context_logs = []
        self.timing_logs = []
        
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
                    model_kw_args=model_kwargs
                )
        
        return pipelines
    
    def run_quick_test(self, queries, answers, configs=None):
        if configs is None:
            configs = ["full_context", "no_context", "original_only"]
        
        # Sample a subset of queries for quick testing
        sample_queries = queries.head(self.sample_size).copy()
        
        results = {}
        
        print(f"Running quick test on {len(sample_queries)} queries...")
        
        for config in configs:
            print(f"\nTesting configuration: {config}")
            start_time = time.time()
            
            config_results = self._run_config(sample_queries, config)
            eval_results = self._evaluate_results(config_results, answers)
            
            end_time = time.time()
            config_time = end_time - start_time
            
            results[config] = {
                "results": config_results,
                "evaluation": eval_results,
                "mean_em": eval_results['em'].mean(),
                "mean_f1": eval_results['f1'].mean(),
                "processing_time": config_time,
                "queries_per_second": len(sample_queries) / config_time
            }
            
            self.timing_logs.append({
                'config': config,
                'time': config_time,
                'queries_processed': len(sample_queries),
                'qps': len(sample_queries) / config_time
            })
            
            print(f"Config {config}: EM={results[config]['mean_em']:.3f}, "
                  f"F1={results[config]['mean_f1']:.3f}, "
                  f"Time={config_time:.1f}s, "
                  f"QPS={results[config]['queries_per_second']:.2f}")
        
        return results
    
    def _run_config(self, queries, config):
        pipeline = self.pipelines[config]
        results = []
        
        for _, query_row in tqdm(queries.iterrows(), total=len(queries), desc=f"Processing {config}"):
            qid = query_row['qid']
            query = query_row['query']
            
            try:
                start_time = time.time()
                result = pipeline.search(query)
                process_time = time.time() - start_time
                
                answer = result.iloc[0]['qanswer'] if not result.empty else None
                
                self._log_analysis(qid, query, config, answer, process_time)
                
                results.append({
                    "qid": qid,
                    "query": query,
                    "qanswer": answer,
                    "config": config,
                    "process_time": process_time
                })
                
            except Exception as e:
                print(f"Error processing query {qid}: {e}")
                results.append({
                    "qid": qid,
                    "query": query,
                    "qanswer": None,
                    "config": config,
                    "process_time": 0,
                    "error": str(e)
                })
        
        return pd.DataFrame(results)
    
    def _log_analysis(self, qid, query, config, answer, process_time):
        log_entry = {
            'qid': qid,
            'query': query,
            'config': config,
            'answer': answer is not None,
            'answer_length': len(answer) if answer else 0,
            'process_time': process_time
        }
        self.context_logs.append(log_entry)
    
    def _evaluate_results(self, res, answers):
        df_content = []
        
        for qid in res.qid.unique():
            golden_answers = answers[answers.qid == qid].gold_answer.to_list()
            
            if res[res.qid == qid].qanswer.isnull().values[0]:
                df_content.append([qid, 0.0, 0.0])
                continue
            
            prediction = res[res.qid == qid].qanswer.values[0]
            em_score = pyterrier_rag._measures.ems(prediction, golden_answers)
            f1_list = []
            for a in golden_answers:
                f1_list.append(pyterrier_rag._measures.f1_score(prediction, a))
            f1_score = max(f1_list)
            
            df_content.append([qid, em_score, f1_score])
        
        return pd.DataFrame(df_content, columns=['qid', 'em', 'f1'])
    
    def save_quick_results(self, results, task, retriever_name):
        """Save results from quick test"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        pathlib.Path("./quick_test_results").mkdir(exist_ok=True)
        pathlib.Path("./quick_test_logs").mkdir(exist_ok=True)
        
        # Save detailed results for each config
        for config, data in results.items():
            filename = f"quick_test_{config}_{retriever_name}_{task}_{timestamp}.csv"
            data['results'].to_csv(f"./quick_test_results/{filename}", index=False)
            
            eval_filename = f"quick_test_eval_{config}_{retriever_name}_{task}_{timestamp}.csv"
            data['evaluation'].to_csv(f"./quick_test_results/{eval_filename}", index=False)
        
        # Save summary comparison
        summary_data = []
        for config, data in results.items():
            summary_data.append({
                'config': config,
                'mean_em': data['mean_em'],
                'mean_f1': data['mean_f1'],
                'processing_time': data['processing_time'],
                'queries_per_second': data['queries_per_second']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"quick_test_summary_{retriever_name}_{task}_{timestamp}.csv"
        summary_df.to_csv(f"./quick_test_results/{summary_filename}", index=False)
        
        # Save logs
        log_filename = f"quick_test_logs_{retriever_name}_{task}_{timestamp}.json"
        with open(f"./quick_test_logs/{log_filename}", 'w') as f:
            json.dump({
                'context_logs': self.context_logs,
                'timing_logs': self.timing_logs,
                'sample_size': self.sample_size
            }, f, indent=2)
        
        print(f"\nQuick test results saved with timestamp: {timestamp}")
        print(f"Summary saved to: ./quick_test_results/{summary_filename}")
        
        return summary_df


def load_retriever_quick(ret, task='nq_test', model='r1'):
    """Simplified retriever loading for quick tests"""
    if not pt.java.started():
        pt.java.init()
    
    print(f'Loading retriever {ret} for {task} (quick test mode)...')
    
    if task == 'nq_test':
        artifact = pt.Artifact.from_hf('pyterrier/ragwiki-terrier')
        sparse_index = pt.Artifact.from_hf('pyterrier/ragwiki-terrier')
        bm25 = sparse_index.bm25(include_fields=['docno', 'text', 'title'], threads=5)
        bm25_pipeline = pt.rewrite.tokenise() >> bm25 >> pt.rewrite.reset()
    elif task == 'hotpotqa_dev':
        sparse_index_path = '../get_res/hotpotqa_sparse_index'
        index_ref = pt.IndexRef.of(sparse_index_path)
        sparse_index = pt.IndexFactory.of(index_ref)
        bm25 = pt.terrier.Retriever(sparse_index, wmodel='BM25')
        bm25_pipeline = pt.rewrite.tokenise() >> bm25 >> sparse_index.text_loader(["text", "title"]) >> pt.rewrite.reset()
    
    if ret == 'bm25':
        return bm25_pipeline
    elif ret == 'monoT5':
        from pyterrier_t5 import MonoT5ReRanker
        monoT5 = MonoT5ReRanker(batch_size=64, verbose=False)
        return (bm25_pipeline % 20) >> monoT5
    elif ret == 'E5':
        from pyterrier_dr import E5
        import pyterrier_dr
        
        if task == 'nq_test':
            e5_index = pt.Artifact.from_hf('pyterrier/ragwiki-e5.flex')
        elif task == 'hotpotqa_dev':
            e5_index = pyterrier_dr.FlexIndex('../get_res/e5_hotpotqa_wiki_index_2.flex')
        
        e5_query_encoder = E5()
        e5_ret = e5_query_encoder >> e5_index.torch_retriever(fp16=True, num_results=120) >> sparse_index.text_loader(["text", "title"])
        return e5_ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", type=str, default='bm25', choices=['bm25', 'monoT5', 'E5'])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--model", type=str, default='r1', choices=['r1', 'r1s'])
    parser.add_argument("--task", type=str, default='nq_test', choices=['nq_test', 'hotpotqa_dev'])
    parser.add_argument("--sample_size", type=int, default=20, help="Number of queries to test")
    parser.add_argument("--configs", type=str, nargs='+', default=['full_context', 'no_context', 'original_only'])
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducible sampling")
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    np.random.seed(args.random_seed)
    
    # Load data
    if args.task == 'nq_test':
        queries = pt.get_dataset('rag:nq').get_topics('test')
        answers = pt.get_dataset('rag:nq').get_answers('test')
    elif args.task == 'hotpotqa_dev':
        queries = pd.read_csv('./hotpotqa_materials/hotpotqa_queries.csv')
        answers = pd.read_csv('./hotpotqa_materials/hotpotqa_answers.csv')
    
    # Shuffle queries for random sampling
    queries = queries.sample(n=min(len(queries), args.sample_size * 3), random_state=args.random_seed).reset_index(drop=True)
    
    # Load retriever
    retriever = load_retriever_quick(args.retriever, args.task, args.model)
    
    # Create quick test pipeline
    quick_tester = QuickTestPipeline(
        retriever=retriever,
        top_k=args.k,
        model=args.model,
        sample_size=args.sample_size
    )
    
    # Run quick test
    print(f"Starting quick test with {args.sample_size} queries...")
    print(f"Configurations to test: {args.configs}")
    
    results = quick_tester.run_quick_test(queries, answers, args.configs)
    
    # Save results
    summary = quick_tester.save_quick_results(results, args.task, args.retriever)
    
    # Print summary
    print("\n" + "="*60)
    print("QUICK TEST SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
    
    # Print timing analysis
    print("\n" + "="*60)
    print("TIMING ANALYSIS")
    print("="*60)
    for log in quick_tester.timing_logs:
        print(f"{log['config']}: {log['time']:.1f}s ({log['qps']:.2f} queries/sec)")
    
    # Estimate full dataset time
    total_queries = len(pt.get_dataset('rag:nq').get_topics('test')) if args.task == 'nq_test' else len(pd.read_csv('./hotpotqa_materials/hotpotqa_queries.csv'))
    
    print(f"\nEstimated time for full dataset ({total_queries} queries):")
    for config, data in results.items():
        estimated_time = (total_queries / data['queries_per_second']) / 60  # in minutes
        print(f"  {config}: ~{estimated_time:.1f} minutes")


if __name__ == "__main__":
    main()