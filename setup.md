# 测试环境

| 资源组    | 3090Group |
|---------|-----------|
| 加速卡系列  | GPU       |
| 加速卡类型  | 3090      |
| CPU核心数 | 4         |
| Python版本 | 3.12      |

# 环境变量设置

```bash
export MODEL_PATH="/zhangguangyi01/Lianghongjian/models"
export DATA_PATH="/zhangguangyi01/Lianghongjian/result"
export CODE_PATH="/zhangguangyi01/Lianghongjian/code"
export LOG_PATH="/zhangguangyi01/Lianghongjian/logs"
```

# 数据集配置

```python
data_config = {
    "max_samples": 1000,               # 最大样本数量
    "min_samples_per_class": 100,      # 每类最小样本数
    "stratification": True,            # 分层抽样
    "strata_fields": ["difficulty"],   # 按难度分层
    
    # 文本处理限制
    "max_prefix_length": 512,          # 前缀最大长度
    "max_suffix_length": 200           # 后缀最大长度
}
```