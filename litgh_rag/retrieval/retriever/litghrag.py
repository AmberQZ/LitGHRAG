import networkx as nx
import json
from tqdm import tqdm
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
import json_repair
from litgh_rag.retrieval.embedding_model import BaseEmbeddingModel
from litgh_rag.reader.llm_generator import LLMGenerator
from logging import Logger
from dataclasses import dataclass
from typing import Optional
from litgh_rag.retrieval.retriever.base import BasePassageRetriever, InferenceConfig
from sentence_transformers import CrossEncoder

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val


class LitGHRAGRetriever(BasePassageRetriever):
    
    def __init__(self, llm_generator:LLMGenerator, 
                 sentence_encoder:BaseEmbeddingModel, 
                 data : dict, 
                 inference_config: Optional[InferenceConfig] = None,
                 logger = None,
                 **kwargs):
        self.rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')        
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder
        self.node_embeddings = data["node_embeddings"]
        self.node_list = data["node_list"]
        self.edge_list = data["edge_list"]
        self.edge_embeddings = data["edge_embeddings"]
        self.text_embeddings = data["text_embeddings"]
        self.edge_faiss_index = data["edge_faiss_index"]
        self.passage_dict = data["text_dict"]
        self.text_id_list = list(self.passage_dict.keys())
        self.KG = data["KG"]
        self.KG = self.KG.subgraph(self.node_list + self.text_id_list)

        self.logger = logger
        if self.logger is None:
            self.logging = False
        else:
            self.logging = True
        
        LitGHRAGmode = "query2edge"
        if LitGHRAGmode == "query2edge":
            self.retrieve_node_fn = self.query2edge
        elif LitGHRAGmode == "query2node":
            self.retrieve_node_fn = self.query2node
        elif LitGHRAGmode == "ner2node":
            self.retrieve_node_fn = self.ner2node
        else:
            raise ValueError(f"Invalid mode: {LitGHRAGmode}. Choose from 'query2edge', 'query2node', or 'query2passage'.")

        self.inference_config = inference_config if inference_config is not None else InferenceConfig()
        node_id_to_file_id = {}
        for node_id in tqdm(list(self.KG.nodes)):
            if self.inference_config.keyword == "musique" and self.KG.nodes[node_id]['type']=="passage":
                node_id_to_file_id[node_id] = self.KG.nodes[node_id]["id"]
            else:
                node_id_to_file_id[node_id] = self.KG.nodes[node_id].get("file_id", node_id)
        self.node_id_to_file_id = node_id_to_file_id

    def ner(self, text):
        return self.llm_generator.ner(text)
    
    def ner2node(self, query, topN = 10):
        entities = self.ner(query)
        entities = entities.split(", ")

        if len(entities) == 0:
            entities = [query]

        topk_nodes = []
        node_score_dict = {}
        for entity_index, entity in enumerate(entities):
            topk_for_this_entity = 5  #控制每个实体取几个节点
            entity_embedding = self.sentence_encoder.encode([entity], query_type="search")
            scores = min_max_normalize(self.node_embeddings@entity_embedding[0].T)
            index_matrix = np.argsort(scores)[-topk_for_this_entity:][::-1]
            similarity_matrix = [scores[i] for i in index_matrix]
            for index, sim_score in zip(index_matrix, similarity_matrix):
                node = self.node_list[index]
                if node not in topk_nodes:
                    topk_nodes.append(node)
                    node_score_dict[node] = sim_score
                    
        topk_nodes = list(set(topk_nodes))
        result_node_score_dict = {}
        for node in topk_nodes:
          if node in node_score_dict:
            result_node_score_dict[node] = node_score_dict[node]
        return result_node_score_dict
    
    def query2node(self, query, topN = 10):
        query_emb = self.sentence_encoder.encode([query], query_type="entity")
        scores = min_max_normalize(self.node_embeddings@query_emb[0].T)
        index_matrix = np.argsort(scores)[-topN:][::-1]
        similarity_matrix = [scores[i] for i in index_matrix]
        result_node_score_dict = {}
        for index, sim_score in zip(index_matrix, similarity_matrix):
            node = self.node_list[index]
            result_node_score_dict[node] = sim_score

        return result_node_score_dict
    
    def query2edge(self, query, topN = 10):
        query_emb = self.sentence_encoder.encode([query], query_type="edge")
        scores = min_max_normalize(self.edge_embeddings@query_emb[0].T)
        index_matrix = np.argsort(scores)[-10:][::-1]
        log_edge_list = []
        for index in index_matrix:
            edge = self.edge_list[index]
            edge_str = [self.KG.nodes[edge[0]]['id'], self.KG.edges[edge]['relation'], self.KG.nodes[edge[1]]['id']]
            log_edge_list.append(edge_str)

        similarity_matrix = [scores[i] for i in index_matrix]
        # construct the edge list
        before_filter_edge_json = {}
        before_filter_edge_json['fact'] = []
        for index, sim_score in zip(index_matrix, similarity_matrix):
            edge = self.edge_list[index]
            edge_str = [self.KG.nodes[edge[0]]['id'], self.KG.edges[edge]['relation'], self.KG.nodes[edge[1]]['id']]
            before_filter_edge_json['fact'].append(edge_str)
        if self.logging:
            self.logger.info(f"LitGHRAG Before Filter Edge: {before_filter_edge_json['fact']}")
        filtered_facts =before_filter_edge_json['fact']
        if len(filtered_facts) == 0:
            return {}, scores
        # use filtered facts to get the edge id and check if it exists in the original candidate list.
        node_score_dict = {}
        log_edge_list = []
        for edge in filtered_facts:
            edge_str = f'{edge[0]} {edge[1]} {edge[2]}'
            search_emb = self.sentence_encoder.encode([edge_str], query_type="search")
            D, I = self.edge_faiss_index.search(search_emb, 1)
            filtered_index = I[0][0]
            # get the edge and the original score
            edge = self.edge_list[filtered_index]
            log_edge_list.append([self.KG.nodes[edge[0]]['id'], self.KG.edges[edge]['relation'], self.KG.nodes[edge[1]]['id']])
            head, tail = edge[0], edge[1]
            sim_score = scores[filtered_index]
            
            if head not in node_score_dict:
                node_score_dict[head] = [sim_score]
            else:
                node_score_dict[head].append(sim_score)
            if tail not in node_score_dict:
                node_score_dict[tail] = [sim_score]
            else:
                node_score_dict[tail].append(sim_score)
        # average the scores
        if self.logging:
            self.logger.info(f"LitGHRAG: Filtered edges: {log_edge_list}")      
        # take average of the scores
        for node in node_score_dict:
            node_score_dict[node] = sum(node_score_dict[node]) / len(node_score_dict[node])
        
        return node_score_dict, scores
    
    def query2passage(self, query, weight_adjust = 0.05):
        query_emb = self.sentence_encoder.encode([query], query_type="passage")
        sim_scores = self.text_embeddings @ query_emb[0].T
        sim_scores = min_max_normalize(sim_scores)*weight_adjust # converted to probability
        # create dict of passage id and score
        return dict(zip(self.text_id_list, sim_scores))
    
    def retrieve_personalization_dict(self, query, topN=30, weight_adjust=0.05):
        node_dict,scores = self.query2edge(query, topN=topN)
        text_dict = self.query2passage(query, weight_adjust=weight_adjust)
        return node_dict, text_dict, scores

    
    def graph_path(self, subG, query, ner_nodes, edge_scores, max_hop=3):
        """
        1. 以ner2node得到的entity节点为起点，只在entity_nodes子图中枚举所有简单路径（不含passage节点）。
        2. 路径分数为每条路径上所有边的embedding分数，首跳全权重，后续每跳权重减半（即第i跳乘0.5**(i-1)），最后累加。
        3. 打印所有路径及其得分。
        返回: [(path, score), ...]
        """
        import networkx as nx

        entity_nodes = [n for n, d in subG.nodes(data=True) if d.get('type') == 'entity']
        entity_sg = subG.subgraph(entity_nodes).copy()
        start_nodes = [n for n in ner_nodes if n in entity_sg.nodes]
        edge_score_dict = {edge: edge_scores[i] for i, edge in enumerate(self.edge_list) if edge in entity_sg.edges}
        # 路径枚举与打分
        all_paths = []
        for s in start_nodes:
            for t in entity_nodes:
                if s == t:
                    continue
                if nx.has_path(entity_sg, s, t):
                    for path in nx.all_simple_paths(entity_sg, s, t, cutoff=max_hop):
                        # 计算路径分数
                        score = 0.0
                        for i in range(len(path)-1):
                            edge = (path[i], path[i+1])
                            edge_score = edge_score_dict.get(edge, 0.0)
                            score += edge_score * (0.5 ** i)
                        all_paths.append((path, score))
        # 按分数排序
        all_paths.sort(key=lambda x: x[1], reverse=True)
        from collections import defaultdict
        top_paths_by_start = defaultdict(list)
        for path, score in all_paths:
            s = path[0]
            top_paths_by_start[s].append((path, score))
        final_paths = []
        for s, paths in top_paths_by_start.items():
            paths.sort(key=lambda x: x[1], reverse=True)
            final_paths.extend(paths[:3])
        result = []
        all_triples = set()  
        for path, score in final_paths:
            # 1. 节点id到文本
            node_texts = []
            for n in path:
                node_info = self.KG.nodes[n]
                node_text = node_info.get('name') or node_info.get('id') or str(n)
                node_texts.append(node_text)
            # 2. 构建(e, r, e)三元组列表
            triples = []
            for i in range(len(path)-1):
                n1, n2 = path[i], path[i+1]
                rel = self.KG.edges[n1, n2].get('relation', '') if self.KG.has_edge(n1, n2) else ''
                triple = (node_texts[i], rel, node_texts[i+1])
                triples.append(triple)
                all_triples.add(triple)
            result.append({
                'path': path,
                'score': score,
                'node_texts': node_texts,
                'rel_path': str(triples)
            })
        all_nodes = set()
        for item in result:
            all_nodes.update(item['path'])
        passage_count = {}
        # 新增：统计每个entity节点被哪些event节点指向
        entity_event_count = {}  # entity_id: [event_id, ...]
        for n in all_nodes:
            n_type = subG.nodes[n].get('type')
            if n_type == 'entity':
                # 找所有前驱event节点
                for pred in subG.predecessors(n):
                    pred_type = subG.nodes[pred].get('type')
                    if pred_type == 'event':
                        entity_event_count.setdefault(n, []).append(pred)
            # 统计passage节点被指向次数
            for nbr in subG.successors(n):
                nbr_type = subG.nodes[nbr].get('type')
                if nbr_type == 'passage':
                    passage_count[nbr] = passage_count.get(nbr, 0) + 1
        # passage排序
        sorted_passages = sorted(passage_count.items(), key=lambda x: x[1], reverse=True)
        passages_contents = []
        passage_ids = []
        for pid, count in sorted_passages:
            text = self.passage_dict.get(pid, '')
            file_id = self.node_id_to_file_id.get(pid, pid)
            passages_contents.append(text)
            passage_ids.append(file_id)
        # 统计event节点指向entity的次数
        event_count = {}  # event_id: count
        for n in all_nodes:
            n_type = subG.nodes[n].get('type')
            if n_type == 'entity':
                for pred in subG.predecessors(n):
                    pred_type = subG.nodes[pred].get('type')
                    if pred_type == 'event':
                        event_count[pred] = event_count.get(pred, 0) + 1
        # 按指向次数递减排序event内容
        sorted_events = sorted(event_count.items(), key=lambda x: x[1], reverse=True)
        events_contents = []
        for eid, count in sorted_events:
            node_info = subG.nodes[eid]
            event_text = node_info.get('name') or node_info.get('id') or str(eid)
            events_contents.append(event_text)

        return {
            'paths': result,
            'passages_contents': passages_contents,
            'passage_ids': passage_ids,
            'passage_count': sorted_passages,
            'events_contents': events_contents,
            'event_count': sorted_events,
            'triples': list(all_triples)
        }

    def reason_path(self, subG, query, ner_nodes, edge_scores, topN=5, max_hop=3):
        """
        对subG只保留entity和passage节点，枚举entity到passage的所有简单路径，按路径长度和节点分数加权，输出高分passage及reason path。
        """
        import networkx as nx
        # 1. 只保留entity和passage节点
        nodes_to_keep = [n for n, d in subG.nodes(data=True) if d.get('type') in ('entity', 'passage')]
        sg = subG.subgraph(nodes_to_keep).copy()
        entity_nodes = [n for n, d in sg.nodes(data=True) if d.get('type') == 'entity']
        passage_nodes = [n for n, d in sg.nodes(data=True) if d.get('type') == 'passage']
        subpath=self.graph_path(subG, query, ner_nodes,edge_scores, max_hop=max_hop)

        # 3. 路径枚举与加权
        path_scores = {}
        path_details = {}
        for p in passage_nodes:
            best_score = 0
            best_path = None
            for e in entity_nodes:
                if nx.has_path(sg, e, p):
                    for path in nx.all_simple_paths(sg, e, p, cutoff=max_hop):
                        node_score = 0
                        for n in path:
                            node_score += sg.degree(n)
                        score = node_score / len(path)
                        if score > best_score:
                            best_score = score
                            best_path = path
            if best_score > 0 and best_path is not None:
                path_scores[p] = best_score
                path_details[p] = best_path
        # 4. 排序输出
        sorted_passages = sorted(path_scores.items(), key=lambda x: x[1], reverse=True)[:topN]
        sorted_passages_contents = [self.passage_dict[pid] for pid, _ in sorted_passages]
        sorted_passage_ids = [self.node_id_to_file_id[pid] for pid, _ in sorted_passages]
        reason_paths = [path_details[pid] for pid, _ in sorted_passages]
        return sorted_passages_contents, sorted_passage_ids, reason_paths, subpath
    

    def retrieve(self, query, topN=5, vner=3, vq2node=10, vq2edge=20, use_reason_path=True, **kwargs):
        """
        1. 用ner2node、query2node、query2edge三种方法分别取前10个节点，交集为seed，并集为C0
        2. 以C0为基础，取两跳邻居，得到C1
        3. 在C1子图上用query2edge的分数做personalization，seed为交集，PPR排序passage
        4. 返回: sorted_passages_contents, sorted_passage_ids
        """
        rerank_model = self.rerank_model
        weight_adjust = self.inference_config.weight_adjust
        
        edge_score_dict, text_dict, edge_scores = self.retrieve_personalization_dict(query, topN=vq2edge, weight_adjust=weight_adjust)
        ner_nodes_dict = self.ner2node(query, topN=vner)
     
        ner_nodes = list(ner_nodes_dict.keys())
        q2n_nodes = list(self.query2node(query, topN=vq2node).keys())
        q2e_nodes = list(edge_score_dict.keys())
        c0_nodes = list(set(ner_nodes) | set(q2n_nodes) | set(q2e_nodes))

        # 2. 子图拓扑扩展，两跳邻居
        c1_nodes = set(c0_nodes)
        for node in c0_nodes:
            neighbors1 = set(self.KG.neighbors(node))
            c1_nodes.update(neighbors1)
            for n1 in neighbors1:
                c1_nodes.update(self.KG.neighbors(n1))
        c1_nodes = list(c1_nodes)
        # 只保留子图中的节点
        subG = self.KG.subgraph(c1_nodes).copy()

        if use_reason_path:
          sorted_passages_contents, sorted_passage_ids, reason_paths, subpath = self.reason_path(subG, query, ner_nodes, edge_scores, topN=topN)


        personalization_dict = {}
        if len(edge_score_dict) == 0:
            print("edge_score_dict is empty, return query2passage results")
            # return topN text passages
            sorted_passages = sorted(text_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_passages = sorted_passages[:topN]
            sorted_passages_contents = []
            sorted_scores = []
            sorted_passage_ids = []
            for passage_id, score in sorted_passages:
                sorted_passages_contents.append(self.passage_dict[passage_id])
                sorted_scores.append(float(score))
                sorted_passage_ids.append(self.node_id_to_file_id[passage_id])

        
        else:            
            personalization_dict.update(edge_score_dict)
            personalization_dict.update(text_dict)
            # 4. PPR on subG
            pr = nx.pagerank(subG, personalization=personalization_dict,
                            alpha=self.inference_config.ppr_alpha,
                            max_iter=self.inference_config.ppr_max_iter,
                            tol=self.inference_config.ppr_tol)
            # 只取passage节点
            text_dict_score = {}
            for node in self.text_id_list:
                if node in pr and pr[node] > 0.0:
                    text_dict_score[node] = pr[node]
            sorted_passages_ids = sorted(text_dict_score.items(), key=lambda x: x[1], reverse=True)
            sorted_passages_ids = sorted_passages_ids[:topN]
            sorted_passages_contents = []
            sorted_scores = []
            sorted_passage_ids = []
            for passage_id, score in sorted_passages_ids:
                sorted_passages_contents.append(self.passage_dict[passage_id])
                sorted_scores.append(score)
                sorted_passage_ids.append(self.node_id_to_file_id[passage_id])

        # 合并subpath和当前检索结果，使用CrossEncoder重排
        sub_contents = subpath.get('passages_contents', [])[:topN]
        sub_ids = subpath.get('passage_ids', [])[:topN]

        # 合并去重
        all_pairs = list(zip(sorted_passages_contents, sorted_passage_ids)) + list(zip(sub_contents, sub_ids))
        # 按id去重
        seen = set()
        merged = []
        for content, pid in all_pairs:
            if pid not in seen:
                merged.append((content, pid))
                seen.add(pid)
        # 用CrossEncoder重排
        # 构造query-passage对
        rerank_inputs = [(query, content) for content, pid in merged]
        scores = rerank_model.predict(rerank_inputs)
        # 按分数降序排序
        reranked = sorted(zip(merged, scores), key=lambda x: x[1], reverse=True)
        final_contents = [pair[0][0] for pair in reranked[:topN]]
        final_ids = [pair[0][1] for pair in reranked[:topN]]
        events = subpath.get('events_contents', [])
        triples = subpath.get('triples', [])
        return final_contents, final_ids, triples, events


