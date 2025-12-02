import os
import sys
import json
import torch
import traceback
from pathlib import Path
from datetime import datetime
import warnings

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')

from transformers import AutoModelForCausalLM, AutoTokenizer

class MemorizationTester:
    def __init__(self, model_path, data_path):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.model = None
        self.tokenizer = None
        self.device = None
        self.test_data = []
        self.results = []
        self.stats = {
            'total_samples': 0,
            'success_count': 0,
            'failure_count': 0,
            'exact_match_count': 0,
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
            
            # 获取模型设备
            self.device = self.model.device
            print(f"Llama模型加载成功，运行在设备: {self.device}")
            return True

        except Exception as e:
            print(f"模型加载失败: {e}")
            traceback.print_exc()
            return False
    
    def load_test_data(self):
        try:
            # 检查数据路径是否存在
            if not self.data_path.exists():
                print(f"数据路径不存在")
                return False
            
            # 如果是目录，查找其中的JSON文件
            data_file = None
            if self.data_path.is_dir():
                # 查找目录中的JSON文件
                json_files = list(self.data_path.glob("*.json"))
                if not json_files:
                    print(f"数据目录中没有找到JSON文件")
                    return False
                # 使用第一个找到的JSON文件
                data_file = json_files[0]
                print(f"找到数据文件: {data_file.name}")
            else:
                # 如果是文件，直接使用
                data_file = self.data_path
            
            # 读取JSON数据
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据结构
            if not isinstance(data, list):
                print(f"数据文件格式错误")
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
            print(f"数据加载失败: {e}")
            return False
    
    def format_prompt(self, prefix_tokens):
        # 将前缀tokens组合成文本
        prefix_text = ' '.join(prefix_tokens)
        
        # 构建提示词（简单的续写指令）
        prompt = f"\n{prefix_text}\n"
        
        return prompt
    
    def preprocess_text(self, text):
        """预处理文本，处理特殊字符和格式"""
        # 处理特殊字符和格式
        # 1. 处理Markdown格式
        text = text.replace('**', ' ')  # 移除Markdown加粗格式
        
        # 2. 处理转义字符
        text = text.replace('\\[', '[')  # 恢复转义的左方括号
        text = text.replace('\\]', ']')  # 恢复转义的右方括号
        text = text.replace('\\{', '{')  # 恢复转义的左花括号
        text = text.replace('\\}', '}')  # 恢复转义的右花括号
        
        # 3. 处理反引号
        text = text.replace('`', ' ')  # 移除反引号
        
        # 4. 处理换行符
        text = text.replace('\n', ' ')  # 换行符替换为空格
        
        # 5. 处理多余的空格
        text = ' '.join(text.split())  # 合并多个空格为一个
        
        return text
    
    def generate_text(self, prompt, max_new_tokens):
        """根据提示词生成文本"""
        try:
            # 使用tokenizer对提示词进行编码
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            # 确保设备已初始化
            if self.device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 使用贪婪解码生成文本
            # 调整生成长度，确保有足够空间生成与参考文本长度匹配的内容
            adjusted_max_tokens = max_new_tokens * 2  # 使用传入的max_new_tokens参数，生成长度翻倍，确保足够的生成空间
            
            output = self.model.generate(
                **inputs,
                max_new_tokens=adjusted_max_tokens,  # 使用调整后的生成长度
                do_sample=False,  # 关闭采样，使用贪婪
                temperature=None,  # 显式设置为None，避免警告
                top_p=None,  # 显式设置为None，避免警告
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=None  # 不使用EOS标记，避免提前终止生成
            )
            
            # 解码生成的文本
            generated_text = self.tokenizer.decode(
                output[0][inputs['input_ids'].shape[1]:],  # 只获取新生成的部分
                skip_special_tokens=True
            )
            
            return generated_text
            
        except Exception as e:
            print(f"文本生成失败: {e}")
            return None
    
    def remove_punctuation(self, tokens):
        if not tokens:
            return []
        
        # 定义常见的标点符号集合
        punctuation = {'.', ',', ';', ':', '!', '?', '(', ')', '[', ']', '{', '}', '"', '\'', '-', '_', '=', '+', '*', '/', '\\', '`', '~', '<', '>'}
        
        # 改进的标点符号处理：处理作为token一部分的标点符号
        filtered_tokens = []
        for token in tokens:
            # 去除token开头和结尾的标点符号
            cleaned_token = token
            while cleaned_token and cleaned_token[0] in punctuation:
                cleaned_token = cleaned_token[1:]
            while cleaned_token and cleaned_token[-1] in punctuation:
                cleaned_token = cleaned_token[:-1]
            # 如果处理后token不为空，则保留
            if cleaned_token:
                filtered_tokens.append(cleaned_token)
        
        return filtered_tokens
    
    def test_memorization(self, sample_size=None):
        """测试模型的记忆能力"""
        print("\n=== 开始记忆能力测试 ===")
        
        if not self.model or not self.tokenizer:
            print("模型未加载")
            return False
        
        if not self.test_data:
            print("测试数据为空")
            return False
        
        # 设置测试样本数量
        test_size = sample_size if sample_size and sample_size > 0 else len(self.test_data)
        test_size = min(test_size, len(self.test_data))
        
        # 重置结果和统计信息
        self.results = []
        self.stats['start_time'] = datetime.now()
        self.stats['success_count'] = 0
        self.stats['failure_count'] = 0
        self.stats['exact_match_count'] = 0
        
        # 开始测试
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
                
                # 3. 预处理和分词生成的文本
                preprocessed_generated = self.preprocess_text(generated_text)
                generated_tokens = preprocessed_generated.split()
                
                # 4. 预处理参考文本
                reference_text = ' '.join(sample['suffix_tokens'])
                preprocessed_reference = self.preprocess_text(reference_text)
                reference_tokens = preprocessed_reference.split()
                
                # 5. 去除标点符号后检查是否完全匹配
                is_exact_match = False
                
                # 去除标点符号
                filtered_generated = self.remove_punctuation(generated_tokens)
                filtered_reference = self.remove_punctuation(reference_tokens)
                
                # 只比较与参考文本相同长度的部分，避免生成长度过长影响匹配
                # 取生成文本和参考文本中较短的长度
                min_length = min(len(filtered_generated), len(filtered_reference))
                
                # 检查去除标点符号后的前min_length个token是否完全匹配
                if min_length == len(filtered_reference):  # 确保生成文本至少与参考文本一样长
                    is_exact_match = all(g == r for g, r in zip(filtered_generated[:min_length], filtered_reference[:min_length]))
                
                if is_exact_match:
                    self.stats['exact_match_count'] += 1
                
                # 5. 处理生成文本，确保与参考文本长度匹配
                reference_length = len(sample['suffix_tokens'])
                    
                # 截断或扩展生成文本，使其与参考文本长度匹配
                if len(generated_tokens) > reference_length:
                    # 截断生成的tokens列表
                    truncated_tokens = generated_tokens[:reference_length]
                    # 重新组合成文本
                    truncated_text = ' '.join(truncated_tokens)
                else:
                    # 生成文本长度不足，使用原始文本
                    truncated_text = generated_text
                    truncated_tokens = generated_tokens
                
                # 6. 保存测试结果（包含所有需要的信息，使用截断后的生成文本）
                test_result = {
                    'id': sample['id'],
                    'prefix_text': ' '.join(prefix_tokens),
                    'suffix_text': ' '.join(sample['suffix_tokens']),
                    'generated_text': truncated_text,  # 使用截断后的文本
                    #'generated_tokens': truncated_tokens,  # 可选：保存截断后的tokens
                    'is_exact_match': is_exact_match
                }
                
                self.results.append(test_result)
                self.stats['success_count'] += 1
                
            except Exception as e:
                self.stats['failure_count'] += 1
                continue
        
        # 更新统计信息
        self.stats['end_time'] = datetime.now()
        
        # 显示测试总结
        print("\n=== 测试结果汇总 ===")
        print(f"总测试样本数: {test_size}")
        print(f"成功: {self.stats['success_count']}, 失败: {self.stats['failure_count']}")
        print(f"精确匹配数: {self.stats['exact_match_count']}")
        print(f"精确匹配率: {self.stats['exact_match_count'] / self.stats['success_count'] * 100:.2f}%" if self.stats['success_count'] > 0 else "精确匹配率: 0.00%")
        
        # 保存测试结果
        self.save_results()
        
        return True
    
    def save_results(self):
        """保存测试结果到JSON文件，并将完全匹配和未完全匹配的数据分别导出"""
        try:
            # 生成结果文件名（包含时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
            # 使用用户指定的结果保存路径
            results_dir = Path("/zhangguangyi01/Lianghongjian/result")
            try:
                # 确保结果目录存在
                results_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"创建结果目录失败: {e}")
                # 尝试使用当前目录作为备选
                results_dir = Path("./results")
                results_dir.mkdir(parents=True, exist_ok=True)
                print(f"使用备选结果目录: {results_dir}")
                
            # 主结果文件
            results_file = results_dir / f"llama_results_{timestamp}.json"
            
            # 准备保存的数据
            save_data = {
                'model': 'Llama',
                'test_type': 'complete_dataset',
                'test_date': datetime.now().isoformat(),
                'test_config': {
                    'total_samples': self.stats['total_samples'],
                    'successful_samples': self.stats['success_count'],
                    'prefix_length': 20
                },
                'stats': {
                    'exact_match_count': self.stats['exact_match_count'],
                    'exact_match_rate': self.stats['exact_match_count'] / self.stats['success_count'] if self.stats['success_count'] > 0 else 0.0
                }
            }
            
            # 保存主结果文件
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n测试结果已保存到: {results_file}")
            
            # 分离完全匹配和未完全匹配的结果
            exact_match_results = [result for result in self.results if result['is_exact_match']]
            non_exact_match_results = [result for result in self.results if not result['is_exact_match']]
            
            # 保存完全匹配的数据
            exact_match_file = results_dir / f"llama_exact_match_results_{timestamp}.json"
            with open(exact_match_file, 'w', encoding='utf-8') as f:
                json.dump(exact_match_results, f, ensure_ascii=False, indent=2)
            
            print(f"完全匹配的结果已保存到: {exact_match_file}")
            print(f"完全匹配的样本数量: {len(exact_match_results)}")
            
            # 保存未完全匹配的数据
            non_exact_match_file = results_dir / f"llama_non_exact_match_results_{timestamp}.json"
            with open(non_exact_match_file, 'w', encoding='utf-8') as f:
                json.dump(non_exact_match_results, f, ensure_ascii=False, indent=2)
            
            print(f"未完全匹配的结果已保存到: {non_exact_match_file}")
            print(f"未完全匹配的样本数量: {len(non_exact_match_results)}")
            
            return results_file
            
        except Exception as e:
            print(f"保存测试结果失败: {e}")
            return None


if __name__ == "__main__":
    # 示例用法
    print("Llama模型记忆能力测试脚本")
    
    # 根据用户提供的路径配置
    MODEL_PATH = "/zhangguangyi01/Lianghongjian/model_Llama"  # 模型路径
    DATA_PATH = "/zhangguangyi01/Lianghongjian/data_set/token_processed_data.json"  # 完整的数据文件路径
        
    # 验证路径是否存在
    if not Path(MODEL_PATH).exists():
        print(f"警告: 模型路径不存在: {MODEL_PATH}")
    if not Path(DATA_PATH).exists():
         print(f"警告: 数据路径不存在: {DATA_PATH}")
    
    # 创建测试器实例
    tester = MemorizationTester(MODEL_PATH, DATA_PATH)
    
    try:
        # 1. 加载模型
        model_success = tester.load_model()
        if not model_success:
            print("\n模型加载失败")
            sys.exit(1)
        
        # 2. 加载测试数据
        data_success = tester.load_test_data()
        if not data_success:
            print("\n数据加载失败")
            sys.exit(1)
        
        # 3. 运行记忆能力测试
        print("\n运行记忆能力测试")
        print("=" * 50)
        print("测试配置:")
        print(f"- 前缀长度: 20 tokens")
        print(f"- 参考后缀: 100 tokens (不足则为剩余)")
        print(f"- 生成长度: 与参考后缀相同")
        print(f"- 解码策略: Greedy Decoding")
        print(f"- 总样本数: {len(tester.test_data)}")
        print("=" * 50)
        
        # 直接使用全部样本进行测试
        sample_size = None
        
        # 开始测试
        test_success = tester.test_memorization(sample_size)
        
        if test_success:
            print("\n测试完成！")
        else:
            print("\n测试失败！")
            
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        traceback.print_exc()
        sys.exit(1)
