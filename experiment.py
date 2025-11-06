import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

class MemorizationTester:
    def __init__(self, model_path, data_path):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.model = None
        self.tokenizer = None
        self.results = []
        self.stats = {
            'total_samples': 0,
            'success_count': 0,
            'failure_count': 0,
            'start_time': None,
            'end_time': None
        }

    def load_model(self):
        """安全加载模型"""
        print("=== 加载Qwen2-8B模型 ===")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.model.eval()
            print("✅ 模型加载成功")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False

    def load_test_data(self, sample_size=1000):
        """加载大规模测试数据"""
        print(f"=== 加载测试数据（目标: {sample_size}样本）===")
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            
            # 分层抽样保证代表性
            difficulties = ['Easy', 'Medium', 'Hard']
            test_data = []
            min_per_class = max(100, sample_size // len(difficulties))  # 每类至少100个
            
            for diff in difficulties:
                diff_data = [d for d in all_data if d.get('difficulty') == diff]
                test_data.extend(diff_data[:min_per_class])
            
            # 随机打乱
            np.random.shuffle(test_data)
            actual_samples = min(sample_size, len(test_data))
            
            print(f"✅ 实际加载 {actual_samples} 个测试样本")
            print(f"   难度分布: Easy={sum(1 for d in test_data[:actual_samples] if d['difficulty']=='Easy')}")
            print(f"            Medium={sum(1 for d in test_data[:actual_samples] if d['difficulty']=='Medium')}")
            print(f"            Hard={sum(1 for d in test_data[:actual_samples] if d['difficulty']=='Hard')}")
            return test_data[:actual_samples]
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return []

    def generate_text(self, prompt, max_new_tokens=50):
        """稳健的文本生成"""
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.model.device)
            
            # 修正的生成配置（避免参数冲突）
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=0.1
                )
            
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return full_output[len(prompt):].strip()
        except Exception as e:
            print(f"❌ 生成失败: {str(e)[:100]}...")
            return ""

    def calculate_similarity(self, generated, ground_truth):
        """综合相似度计算"""
        if not generated or not ground_truth:
            return 0.0
        
        # 词重叠相似度
        gen_words = set(generated.lower().split())
        truth_words = set(ground_truth.lower().split())
        if not truth_words:
            return 0.0
        
        word_overlap = len(gen_words & truth_words) / len(truth_words)
        
        # 编辑距离相似度（简化版）
        max_len = max(len(generated), len(ground_truth))
        edit_sim = 1.0 - (abs(len(generated) - len(ground_truth)) / max_len) if max_len > 0 else 0.0
        
        # 综合相似度（加权平均）
        return 0.6 * word_overlap + 0.4 * edit_sim

    def get_similarity_distribution(self, similarity_scores):
        """计算相似度分布"""
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        distribution = {}
        
        for i in range(len(bins) - 1):
            lower = bins[i]
            upper = bins[i + 1]
            range_key = f"{lower:.1f}-{upper:.1f}"
            
            if i == len(bins) - 2:  # 最后一个区间包含上限
                count = sum(lower <= score <= upper for score in similarity_scores)
            else:
                count = sum(lower <= score < upper for score in similarity_scores)
            
            percentage = (count / len(similarity_scores)) * 100 if similarity_scores else 0
            distribution[range_key] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        return distribution

    def test_memorization(self, test_data, max_new_tokens=50):
        """执行记忆率测试"""
        print("\n=== 开始记忆率测试 ===")
        self.stats['start_time'] = datetime.now()
        self.stats['total_samples'] = len(test_data)
        
        exact_matches = 0
        similarity_scores = []
        
        for item in tqdm(test_data, desc="测试进度"):
            try:
                prefix = item['prefix']
                true_suffix = item['true_suffix']
                
                generated = self.generate_text(prefix, max_new_tokens)
                if not generated:
                    self.stats['failure_count'] += 1
                    continue
                
                # 计算指标
                exact_match = (generated.strip() == true_suffix.strip())
                similarity = self.calculate_similarity(generated, true_suffix)
                
                if exact_match:
                    exact_matches += 1
                similarity_scores.append(similarity)
                
                self.results.append({
                    'problem_id': item.get('id'),
                    'difficulty': item.get('difficulty'),
                    'exact_match': exact_match,
                    'similarity': similarity,
                    'generated': generated[:100] + "..." if len(generated) > 100 else generated,
                    'ground_truth': true_suffix[:100] + "..." if len(true_suffix) > 100 else true_suffix
                })
                self.stats['success_count'] += 1
                
            except Exception as e:
                print(f"❌ 测试失败: {str(e)[:100]}...")
                self.stats['failure_count'] += 1
                continue
        
        self.stats['end_time'] = datetime.now()
        return exact_matches, similarity_scores

    def save_results(self):
        """保存测试结果"""
        output_dir = Path("/zhangguangyi01/Lianghongjian/result")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "memorization_results.json"
        
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        # 计算相似度分布
        similarity_scores = [r['similarity'] for r in self.results]
        similarity_distribution = self.get_similarity_distribution(similarity_scores)
        
        results_summary = {
            'model': 'Qwen2-8B',
            'test_date': datetime.now().isoformat(),
            'test_config': {
                'sample_size': self.stats['total_samples'],
                'max_new_tokens': 50,
                'duration_seconds': duration
            },
            'metrics': {
                'exact_match_rate': sum(r['exact_match'] for r in self.results) / len(self.results) if self.results else 0,
                'average_similarity': np.mean([r['similarity'] for r in self.results]) if self.results else 0,
                'success_rate': self.stats['success_count'] / self.stats['total_samples'] if self.stats['total_samples'] > 0 else 0
            },
            'similarity_distribution': similarity_distribution,
            'by_difficulty': {
                diff: {
                    'count': sum(1 for r in self.results if r['difficulty'] == diff),
                    'emr': sum(r['exact_match'] for r in self.results if r['difficulty'] == diff) / max(1, sum(1 for r in self.results if r['difficulty'] == diff)),
                    'avg_similarity': np.mean([r['similarity'] for r in self.results if r['difficulty'] == diff])
                } for diff in ['Easy', 'Medium', 'Hard']
            },
            'detailed_results': self.results[:200]  # 保存前200条详细结果
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存至: {output_path}")
        
        return similarity_distribution

if __name__ == "__main__":
    # 初始化测试器
    tester = MemorizationTester(
        model_path="/zhangguangyi01/Lianghongjian/models",
        data_path="/zhangguangyi01/Lianghongjian/result/processed_leetcode_data.json"
    )
    
    # 1. 加载模型
    if not tester.load_model():
        exit(1)
    
    # 2. 加载测试数据（1000样本）
    test_data = tester.load_test_data(sample_size=1000)
    if not test_data:
        exit(1)
    
    # 3. 执行测试
    exact_matches, similarity_scores = tester.test_memorization(test_data)
    
    # 4. 计算相似度分布
    similarity_distribution = tester.get_similarity_distribution(similarity_scores)
    
    # 5. 打印摘要
    print("\n=== 测试摘要 ===")
    print(f"总样本数: {tester.stats['total_samples']}")
    print(f"成功测试: {tester.stats['success_count']}")
    print(f"精确匹配数: {exact_matches}")
    print(f"平均相似度: {np.mean(similarity_scores) if similarity_scores else 0:.4f}")
    print(f"总耗时: {(tester.stats['end_time'] - tester.stats['start_time']).total_seconds()/60:.1f}分钟")
    
    print("\n=== 相似度分布 ===")
    for range_key, stats in similarity_distribution.items():
        print(f"{range_key}: {stats['count']}个样本 ({stats['percentage']}%)")
    
    # 6. 保存结果
    similarity_distribution = tester.save_results()
