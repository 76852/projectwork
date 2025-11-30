import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

# 设置Transformers日志级别为ERROR，减少不必要的警告
os.environ['TRANSFORMERS_VERBOSITY'] = 'ERROR'
warnings.filterwarnings('ignore')

# 在设置环境变量后再导入Transformers库
from transformers import AutoModelForCausalLM, AutoTokenizer

class MemorizationTester:
    def __init__(self, model_path, data_path):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.model = None
        self.tokenizer = None
        self.test_data = []
        self.results = []
        self.stats = {
            'total_samples': 0,
            'success_count': 0,
            'failure_count': 0,
            'start_time': None,
            'end_time': None
        }

    def load_model(self):
        print("=== 加载Llama模型 ===")
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            
            # Llama模型特殊处理
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16,  # 使用半精度加速
                device_map="auto",  # 自动分配设备
                low_cpu_mem_usage=True,  # 低内存使用模式
                trust_remote_code=True,
                attn_implementation="eager"  # 确保兼容性
            )
            
            # 设置模型为评估模式
            self.model.eval()
            print("✓ Llama模型加载成功")
            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_test_data(self):
        try:
            # 检查数据路径是否存在
            if not self.data_path.exists():
                print(f"❌ 数据路径不存在")
                return False
            
            # 如果是目录，查找其中的JSON文件
            data_file = None
            if self.data_path.is_dir():
                # 查找目录中的JSON文件
                json_files = list(self.data_path.glob("*.json"))
                if not json_files:
                    print(f"❌ 数据目录中没有找到JSON文件")
                    return False
                # 使用第一个找到的JSON文件
                data_file = json_files[0]
                print(f"✓ 找到数据文件: {data_file.name}")
            else:
                # 如果是文件，直接使用
                data_file = self.data_path
            
            # 读取JSON数据
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据结构
            if not isinstance(data, list):
                print(f"❌ 数据文件格式错误")
                return False
            
            # 检查数据项是否包含必要字段
            required_fields = ['id', 'original_description', 'total_tokens', 'prefix_tokens', 'suffix_tokens']
            valid_data = []
            error_count = 0
            
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    error_count += 1
                    continue
                
                # 检查必要字段
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    error_count += 1
                    continue
                
                # 检查字段类型
                if not isinstance(item['total_tokens'], int) or not isinstance(item['prefix_tokens'], list) or not isinstance(item['suffix_tokens'], list):
                    error_count += 1
                    continue
                
                # 验证前缀和后缀的长度要求
                if len(item['prefix_tokens']) < 1 or len(item['suffix_tokens']) < 1:
                    error_count += 1
                    continue
                
                valid_data.append(item)
            
            # 更新测试数据和统计信息
            self.test_data = valid_data
            self.stats['total_samples'] = len(valid_data)
            

            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def format_prompt(self, prefix_tokens):
        """格式化提示词，使用前缀tokens构建续写提示"""
        # 将前缀tokens组合成文本
        prefix_text = ' '.join(prefix_tokens)
        
        # 构建提示词（简单的续写指令）
        prompt = f"\n{prefix_text}\n"
        
        return prompt
    
    def generate_text(self, prompt, max_length):
        """使用贪婪解码策略生成文本"""
        try:
            # 编码提示词
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 使用贪婪解码生成文本
            # 贪婪解码模式下不需要temperature参数
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_length,  # 生成的新token数量
                do_sample=False,  # 关闭采样，使用贪婪
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # 解码生成的文本
            generated_text = self.tokenizer.decode(
                output[0][inputs['input_ids'].shape[1]:],  # 只获取新生成的部分
                skip_special_tokens=True
            )
            
            return generated_text
            
        except Exception as e:
            print(f"❌ 文本生成失败: {e}")
            return None
    
    def calculate_matching_length(self, generated_tokens, reference_tokens):
        """计算匹配长度(ML)：首次出现偏差之前的token数量"""
        if not generated_tokens or not reference_tokens:
            return 0
        
        # 计算最长连续匹配长度
        matching_length = 0
        min_length = min(len(generated_tokens), len(reference_tokens))
        
        # 遍历tokens，直到发现不匹配或遍历结束
        for i in range(min_length):
            if generated_tokens[i] == reference_tokens[i]:
                matching_length += 1
            else:
                break  # 首次出现偏差，停止计数
        
        return matching_length
    
    def calculate_emr(self, generated_tokens, reference_tokens):
        """计算精确匹配率(EMR)：完全匹配的token比例"""
        if not generated_tokens or not reference_tokens:
            return 0.0
        
        # 取较短的长度进行比较
        min_length = min(len(generated_tokens), len(reference_tokens))
        
        # 计算匹配的token数量
        matching_count = sum(1 for g, r in zip(generated_tokens[:min_length], reference_tokens[:min_length]) if g == r)
        
        # 计算准确率
        emr = matching_count / min_length
        
        return emr
    
    def calculate_rouge_l(self, generated_tokens, reference_tokens):
        """计算ROUGE-L分数：最长公共子序列(LCS)比例"""
        if not generated_tokens or not reference_tokens:
            return 0.0
        
        # 使用动态规划计算最长公共子序列(LCS)长度
        m, n = len(generated_tokens), len(reference_tokens)
        
        # 创建DP表格
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 填充DP表格
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if generated_tokens[i - 1] == reference_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        # LCS长度
        lcs_length = dp[m][n]
        
        # ROUGE-L分数计算 (F1分数)
        precision = lcs_length / m if m > 0 else 0
        recall = lcs_length / n if n > 0 else 0
        
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        return f1_score
    
    def calculate_similarity_scores(self, generated_tokens, reference_tokens):
        """计算综合相似度指标，整合三个评估指标"""
        # 计算匹配长度(ML)
        matching_length = self.calculate_matching_length(generated_tokens, reference_tokens)
        
        # 计算精确匹配率(EMR)
        emr = self.calculate_emr(generated_tokens, reference_tokens)
        
        # 计算ROUGE-L分数
        rouge_l = self.calculate_rouge_l(generated_tokens, reference_tokens)
        
        # 返回包含所有指标的字典
        return {
            'matching_length': matching_length,
            'emr': emr,
            'rouge_l': rouge_l,
            'similarity': emr  # 添加similarity字段，与其他模型保持一致
        }
    
    def get_metric_distribution(self, scores, metric_name, bins=None):
        """计算指定指标的分布
        
        Args:
            scores: 指标分数列表
            metric_name: 指标名称（用于定制区间范围）
            bins: 自定义区间边界，默认使用[0, 0.1, 0.2, ..., 1.0]
            
        Returns:
            包含各区间计数和百分比的分布字典
        """
        if not scores:
            return {}
        
        # 根据指标类型设置合适的区间
        if bins is None:
            if metric_name == 'matching_length':
                # 匹配长度使用更宽的区间
                bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            else:
                # EMR、ROUGE-L和相似度使用0-1区间
                bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        distribution = {}
        
        for i in range(len(bins) - 1):
            lower = bins[i]
            upper = bins[i + 1]
            range_key = f"{lower:.1f}-{upper:.1f}"
            
            if i == len(bins) - 2:  # 最后一个区间包含上限
                count = sum(1 for score in scores if lower <= score <= upper)
            else:
                count = sum(1 for score in scores if lower <= score < upper)
            
            percentage = (count / len(scores)) * 100 if scores else 0
            distribution[range_key] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        return distribution
    
    def get_similarity_distribution(self, similarity_scores):
        """计算相似度分布（使用通用分布计算方法）"""
        return self.get_metric_distribution(similarity_scores, 'similarity')
    
    def get_length_based_statistics(self):
        """计算按生成文本长度区间的统计信息"""
        if not self.results:
            return {}
        
        # 定义长度区间（按tokens数）
        # 根据实际数据分布调整区间
        bins = [0, 20, 40, 60, 80, 100, 120, 150, float('inf')]
        length_stats = {}
        
        for i in range(len(bins) - 1):
            lower = bins[i]
            upper = bins[i + 1]
            
            # 生成区间描述
            if upper == float('inf'):
                range_key = f">={lower}"
            else:
                range_key = f"{lower}-{upper-1}"
            
            # 收集该区间内的所有样本
            range_samples = []
            for result in self.results:
                if 'generated_length' in result:
                    length = result['generated_length']
                    if lower <= length < upper or (upper == float('inf') and length >= lower):
                        range_samples.append(result)
            
            if range_samples:
                # 计算统计指标
                count = len(range_samples)
                total_emr = sum(r['similarity_scores']['emr'] for r in range_samples)
                total_ml = sum(r['similarity_scores']['matching_length'] for r in range_samples)
                total_rouge = sum(r['similarity_scores']['rouge_l'] for r in range_samples)
                
                # 计算平均值
                avg_emr = total_emr / count
                avg_ml = total_ml / count
                avg_rouge = total_rouge / count
                
                # 计算百分比
                percentage = (count / len(self.results)) * 100
                
                # 保存该区间的统计信息
                length_stats[range_key] = {
                    'count': count,
                    'percentage': round(percentage, 2),
                    'avg_emr': round(avg_emr, 4),
                    'avg_matching_length': round(avg_ml, 2),
                    'avg_rouge_l': round(avg_rouge, 4)
                }
            # 不保存没有样本的区间（删除else分支）
        
        return length_stats
    
    def test_memorization(self, sample_size=None):
        """测试模型的记忆能力"""
        print("\n=== 开始记忆能力测试 ===")
        
        if not self.model or not self.tokenizer:
            print("❌ 模型未加载")
            return False
        
        if not self.test_data:
            print("❌ 测试数据为空")
            return False
        
        # 设置测试样本数量
        test_size = sample_size if sample_size and sample_size > 0 else len(self.test_data)
        test_size = min(test_size, len(self.test_data))
        
        # 重置结果和统计信息
        self.results = []
        self.stats['start_time'] = datetime.now()
        self.stats['success_count'] = 0
        self.stats['failure_count'] = 0
        
        # 开始测试
        total_emr = 0.0
        total_matching_length = 0
        total_rouge_l = 0.0
        
        for idx in range(test_size):
            sample = self.test_data[idx]
            
            try:
                # 1. 格式化提示词（使用前20个token作为前缀）
                prefix_tokens = sample['prefix_tokens'][:20]  # 确保只使用前20个token
                prompt = self.format_prompt(prefix_tokens)
                
                # 2. 生成文本（生成长度与参考后缀相同）
                max_new_tokens = len(sample['suffix_tokens'])  # 与参考后缀相同长度
                generated_text = self.generate_text(prompt, max_new_tokens)
                
                if generated_text is None:
                    self.stats['failure_count'] += 1
                    continue
                
                # 3. 对生成的文本进行分词，以便比较
                generated_tokens = generated_text.split()
                
                # 4. 计算相似度指标
                similarity_scores = self.calculate_similarity_scores(generated_tokens, sample['suffix_tokens'])
                
                # 累加指标用于计算平均值
                total_emr += similarity_scores['emr']
                total_matching_length += similarity_scores['matching_length']
                total_rouge_l += similarity_scores['rouge_l']
                
                # 5. 保存测试结果
                test_result = {
                    'id': sample['id'],
                    'prefix_tokens': prefix_tokens,
                    'reference_suffix_tokens': sample['suffix_tokens'],
                    'generated_text': generated_text,
                    'generated_tokens': generated_tokens,
                    'generated_length': len(generated_tokens),  # 记录生成文本长度
                    'similarity_scores': similarity_scores,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.results.append(test_result)
                self.stats['success_count'] += 1
                
            except Exception as e:
                self.stats['failure_count'] += 1
                continue
        
        # 计算平均相似度指标
        if self.stats['success_count'] > 0:
            avg_emr = total_emr / self.stats['success_count']
            avg_matching_length = total_matching_length / self.stats['success_count']
            avg_rouge_l = total_rouge_l / self.stats['success_count']
        else:
            avg_emr = 0.0
            avg_matching_length = 0
            avg_rouge_l = 0.0
        
        # 更新统计信息
        self.stats['end_time'] = datetime.now()
        self.stats['average_emr'] = avg_emr
        self.stats['average_matching_length'] = avg_matching_length
        self.stats['average_rouge_l'] = avg_rouge_l
        
        # 显示测试总结
        print("\n=== 测试结果汇总 ===")
        print(f"总测试样本数: {test_size}")
        print(f"成功: {self.stats['success_count']}, 失败: {self.stats['failure_count']}")
        
        if self.stats['success_count'] > 0:
            print(f"\n平均指标:")
            print(f"  平均匹配长度(ML): {avg_matching_length:.2f} tokens")
            print(f"  平均精确匹配率(EMR): {avg_emr:.4f}")
            print(f"  平均ROUGE-L: {avg_rouge_l:.4f}")
        
        # 保存完整测试结果
        self.save_complete_results()
        
        return True
    
    def save_results(self):
        """保存测试结果到JSON文件（基础版）"""
        try:
            # 生成结果文件名（包含时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 使用用户指定的结果保存路径
            results_dir = Path("/zhangguangyi01/Lianghongjian/result")
            # 确保结果目录存在
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = results_dir / f"llama_results_{timestamp}.json"
            
            # 准备保存的数据
            save_data = {
                'config': {
                    'model_path': str(self.model_path),
                    'data_path': str(self.data_path),
                    'decoding_strategy': 'greedy',
                    'prefix_length': 20
                },
                'stats': self.stats,
                'results': self.results
            }
            
            # 保存文件
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✓ 测试结果已保存到: {results_file}")
            
            return results_file
            
        except Exception as e:
            print(f"❌ 保存测试结果失败: {e}")
            return None
    
    def save_complete_results(self):
        """保存完整测试结果（与其他模型保持一致的格式）"""
        try:
            # 使用用户指定的结果保存路径
            results_dir = Path("/zhangguangyi01/Lianghongjian/result")
            results_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"llama_results_{timestamp}.json"
            
            # 计算测试耗时
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds() if self.stats['start_time'] and self.stats['end_time'] else 0
            
            # 计算相似度分布
            similarity_scores = [r['similarity_scores']['similarity'] for r in self.results]
            similarity_distribution = self.get_similarity_distribution(similarity_scores)
            
            # 计算其他指标分布
            emr_scores = [r['similarity_scores']['emr'] for r in self.results]
            matching_length_scores = [r['similarity_scores']['matching_length'] for r in self.results]
            rouge_l_scores = [r['similarity_scores']['rouge_l'] for r in self.results]
            
            emr_distribution = self.get_metric_distribution(emr_scores, 'emr')
            ml_distribution = self.get_metric_distribution(matching_length_scores, 'matching_length')
            rouge_l_distribution = self.get_metric_distribution(rouge_l_scores, 'rouge_l')
            
            # 计算生成文本长度区间统计
            length_based_statistics = self.get_length_based_statistics()
            
            # 准备结果摘要，调整输出顺序
            results_summary = {
                'model': 'Llama',
                'test_type': 'complete_dataset',
                'test_date': datetime.now().isoformat(),
                'test_config': {
                    'total_samples': self.stats['total_samples'],
                    'successful_samples': self.stats['success_count'],
                    'prefix_length': 20,
                    'duration_seconds': round(duration, 4)
                },
                'metrics': {
                    'exact_match_rate': round(sum(1 for r in self.results if r['similarity_scores']['emr'] == 1.0) / len(self.results) if self.results else 0, 4),
                    'average_similarity': round(np.mean(similarity_scores) if similarity_scores else 0, 4),
                    'success_rate': round(self.stats['success_count'] / self.stats['total_samples'] if self.stats['total_samples'] > 0 else 0, 4),
                    'average_emr': round(self.stats['average_emr'] if 'average_emr' in self.stats else 0, 4),
                    'average_matching_length': round(self.stats['average_matching_length'] if 'average_matching_length' in self.stats else 0, 4),
                    'average_rouge_l': round(self.stats['average_rouge_l'] if 'average_rouge_l' in self.stats else 0, 4)
                },
                'similarity_distribution': similarity_distribution,
                'emr_distribution': emr_distribution,
                'matching_length_distribution': ml_distribution,
                'rouge_l_distribution': rouge_l_distribution,
                'length_based_statistics': length_based_statistics  # 添加长度区间统计
            }
        
            # 保存文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
            print(f"\n✓ 完整测试结果已保存至: {output_path}")
            return similarity_distribution
            
        except Exception as e:
            print(f"❌ 保存完整测试结果失败: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # 示例用法
    print("Llama模型记忆能力测试脚本")
    
    # 根据用户提供的路径配置
    MODEL_PATH = "/zhangguangyi01/Lianghongjian/model_Llama"  # 模型路径
    DATA_PATH = "/zhangguangyi01/Lianghongjian/data_set/token_processed_data.json"  # 完整的数据文件路径
    
    # 创建测试器实例
    tester = MemorizationTester(MODEL_PATH, DATA_PATH)
    
    try:
        # 1. 加载模型
        print("加载Llama模型...")
        model_success = tester.load_model()
        
        if not model_success:
            print("\n✗ 模型加载失败")
            sys.exit(1)
        
        # 2. 加载测试数据
        print("\n加载测试数据...")
        data_success = tester.load_test_data()
        
        if not data_success:
            print("\n✗ 数据加载失败")
            sys.exit(1)
        
        # 3. 运行记忆能力测试
        print("\n运行记忆能力测试")
        print("=" * 50)
        print("测试配置:")
        print("- 前缀长度: 20 tokens")
        print("- 参考后缀: 100 tokens (不足则为剩余)")
        print("- 生成长度: 与参考后缀相同")
        print("- 解码策略: Greedy Decoding")
        print("=" * 50)
        
        # 直接使用全部样本进行测试
        sample_size = None
        
        # 开始测试
        test_success = tester.test_memorization(sample_size)
            
    except KeyboardInterrupt:
        print("\n\n❌ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()