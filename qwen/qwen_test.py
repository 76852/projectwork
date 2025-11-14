import json
import torch
import numpy as np
from pathlib import Path
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
        """安全加载Qwen2-8B模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                local_files_only=True
            )
            # Qwen模型特殊处理
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                attn_implementation="eager",  # 确保兼容性
                local_files_only=True
            )
            self.model.eval()
            print("Qwen2-8B模型加载成功")
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def load_all_test_data(self):
        """加载数据集中的所有测试数据"""
        print("=== 加载完整测试数据集 ===")
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            # 统计难度分布
            difficulties = ['Easy', 'Medium', 'Hard']
            total_count = len(all_data)
        
            print(f"   加载完整数据集: {total_count} 个样本")
            for diff in difficulties:
                diff_count = sum(1 for d in all_data if d.get('difficulty') == diff)
                print(f"   {diff}: {diff_count} 个样本")
        
            return all_data
        except Exception as e:
            print(f"数据加载失败: {e}")
            return []

    def format_prompt_for_qwen(self, prompt):
        """为Qwen2模型格式化提示词"""
        # Qwen2模型使用特殊的对话格式
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def generate_text(self, prompt, max_new_tokens=50):
        """为Qwen2模型优化的文本生成"""
        try:
            # 格式化提示词
            formatted_prompt = self.format_prompt_for_qwen(prompt)
            
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.model.device)
            
            # Qwen2模型生成配置
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )
            
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 提取模型回复部分（去除指令标签）
            response = full_output.replace(formatted_prompt, "").strip()
            return response
        except Exception as e:
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
        """计算相似度分布 - 修复空列表问题"""
        if not similarity_scores:
            return {}
            
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        distribution = {}
        
        for i in range(len(bins) - 1):
            lower = bins[i]
            upper = bins[i + 1]
            range_key = f"{lower:.1f}-{upper:.1f}"
            
            if i == len(bins) - 2:  # 最后一个区间包含上限
                count = sum(1 for score in similarity_scores if lower <= score <= upper)
            else:
                count = sum(1 for score in similarity_scores if lower <= score < upper)
            
            percentage = (count / len(similarity_scores)) * 100 if similarity_scores else 0
            distribution[range_key] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        return distribution

    def test_complete_memorization(self, max_new_tokens=50):
        """执行完整数据集的记忆率测试（精简日志版）"""
        print("\n=== 开始Qwen2-8B完整数据集记忆率测试 ===")
        self.stats['start_time'] = datetime.now()
        
        # 加载所有数据
        test_data = self.load_all_test_data()
        self.stats['total_samples'] = len(test_data)
        
        if not test_data:
            print("❌❌❌❌❌❌❌❌ 无法加载测试数据")
            return 0, []
    
        exact_matches = 0
        similarity_scores = []
    
        for i, item in enumerate(test_data):
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
                self.stats['failure_count'] += 1
                continue
    
        self.stats['end_time'] = datetime.now()
        return exact_matches, similarity_scores

    def save_complete_results(self):
        """保存完整测试结果并输出测试摘要"""
        output_dir = Path("/zhangguangyi01/Lianghongjian/result")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"qwen2_results_{timestamp}.json"
        
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        # 计算相似度分布
        similarity_scores = [r['similarity'] for r in self.results]
        similarity_distribution = self.get_similarity_distribution(similarity_scores)
        
        # 计算精确匹配数
        exact_matches = sum(r['exact_match'] for r in self.results)
        
        # 输出完整测试结果
        print(f"\n=== 完整数据集测试结果 ===")
        print(f"总样本数: {self.stats['total_samples']}")
        print(f"成功测试: {self.stats['success_count']}")
        print(f"失败测试: {self.stats['failure_count']}")
        print(f"精确匹配数: {exact_matches}")
        
        if self.stats['success_count'] > 0:
            print(f"精确匹配率: {exact_matches/self.stats['success_count']*100:.2f}%")
        else:
            print("精确匹配率: 0.00%")
            
        if similarity_scores:
            print(f"平均相似度: {np.mean(similarity_scores):.4f}")
        else:
            print("平均相似度: 0.0000")
            
        print(f"总耗时: {duration/60:.1f}分钟")
        
        print("\n=== 相似度区间分布 ===")
        if similarity_distribution:
            for interval, stats in similarity_distribution.items():
                # 使用统一的格式，方便bash脚本匹配
                print(f"{interval}: {stats['count']}个样本 ({stats['percentage']}%)")
        else:
            bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            for i in range(len(bins) - 1):
                lower = bins[i]
                upper = bins[i + 1]
                range_key = f"{lower:.1f}-{upper:.1f}"
                print(f"{range_key}: 0个样本 (0.00%)")
        
        print("\n=== 难度级别分析 ===")
        for diff in ['Easy', 'Medium', 'Hard']:
            diff_results = [r for r in self.results if r['difficulty'] == diff]
            if diff_results:
                emr = sum(r['exact_match'] for r in diff_results) / len(diff_results)
                avg_sim = np.mean([r['similarity'] for r in diff_results])
                # 使用统一的格式
                print(f"{diff}: {len(diff_results)}样本, EMR: {emr*100:.2f}%, 相似度: {avg_sim:.4f}")
            else:
                print(f"{diff}: 0样本, EMR: 0.00%, 相似度: 0.0000")
        
        results_summary = {
            'model': 'Qwen/Qwen2-8B-Instruct',
            'test_type': 'complete_dataset',
            'test_date': datetime.now().isoformat(),
            'test_config': {
                'total_samples': self.stats['total_samples'],
                'successful_samples': self.stats['success_count'],
                'max_new_tokens': 50,
                'duration_seconds': duration
            },
            'metrics': {
                'exact_match_rate': sum(r['exact_match'] for r in self.results) / len(self.results) if self.results else 0,
                'average_similarity': np.mean([r['similarity'] for r in self.results]) if self.results else 0,
                'success_rate': self.stats['success_count'] / self.stats['total_samples'] if self.stats['total_samples'] > 0 else 0
            },
            'similarity_distribution': similarity_distribution,
            'difficulty_analysis': {
                diff: {
                    'count': sum(1 for r in self.results if r['difficulty'] == diff),
                    'exact_match_rate': sum(r['exact_match'] for r in self.results if r['difficulty'] == diff) / max(1, sum(1 for r in self.results if r['difficulty'] == diff)),
                    'average_similarity': np.mean([r['similarity'] for r in self.results if r['difficulty'] == diff]) if any(r['difficulty'] == diff for r in self.results) else 0
                } for diff in ['Easy', 'Medium', 'Hard']
            }
        }
    
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
        print(f"\n完整数据集测试结果已保存至: {output_path}")
        return similarity_distribution

if __name__ == "__main__":
    try:
        # 初始化测试器
        tester = MemorizationTester(
            model_path="/zhangguangyi01/Lianghongjian/models",
            data_path="/zhangguangyi01/Lianghongjian/result/processed_leetcode_data.json"
        )
        
        # 1. 加载模型
        if not tester.load_model():
            print("模型加载失败")
            exit(1)
        
        # 2. 执行完整数据集测试
        exact_matches, similarity_scores = tester.test_complete_memorization()
        
        # 4. 保存完整结果（会自动输出测试结果）
        if tester.results:
            similarity_distribution = tester.save_complete_results()
        
        exit(0)  # 正常退出
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        exit(1)  # 错误退出