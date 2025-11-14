#!/bin/bash
# Mistral-7B记忆率测试脚本 - 修复日志输出问题

# ==================== 配置区域 ====================
BASE_DIR="/zhangguangyi01/Lianghongjian"
SCRIPT="${BASE_DIR}/code/mistral_test.py"
LOG_DIR="${BASE_DIR}/logs"
RESULT_DIR="${BASE_DIR}/result"
PYTHON_EXEC="/usr/bin/python3.12"
TIMEOUT_DURATION="4h"
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
# ==================================================

# 创建日志文件
LOG_FILE="${LOG_DIR}/mistral_test_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}"

# ==================== 日志函数 ====================
log() {
    echo "$1" | tee -a "${LOG_FILE}"
}

# ==================== 主执行流程 ====================

log "[INFO] 开始Mistral-7B记忆率测试"
log "[INFO] 启动测试执行"

# 执行测试并捕获输出
TEST_OUTPUT=$(timeout ${TIMEOUT_DURATION} ${PYTHON_EXEC} "${SCRIPT}" 2>&1)
EXIT_CODE=$?

# 记录完整的Python输出到日志文件（不重复显示在控制台）
echo "=== 完整Python输出 ===" >> "${LOG_FILE}"
echo "${TEST_OUTPUT}" >> "${LOG_FILE}"
echo "=== 完整输出结束 ===" >> "${LOG_FILE}"

# 提取关键信息并格式化输出到控制台和日志文件
echo "=== 关键测试结果 ===" | tee -a "${LOG_FILE}"

# 提取数据加载信息
DATA_LOADING_INFO=$(echo "${TEST_OUTPUT}" | grep -E "加载完整数据集.*个样本|Easy:.*个样本|Medium:.*个样本|Hard:.*个样本")
if [ -n "$DATA_LOADING_INFO" ]; then
    echo "$DATA_LOADING_INFO" | while IFS= read -r line; do
        echo "$line" | tee -a "${LOG_FILE}"
    done
fi

# 提取总样本信息
BASIC_STATS=$(echo "${TEST_OUTPUT}" | grep -E "总样本数|成功测试|失败测试|精确匹配数")
if [ -n "$BASIC_STATS" ]; then
    echo "$BASIC_STATS" | while IFS= read -r line; do
        echo "$line" | tee -a "${LOG_FILE}"
    done
fi

# 提取核心指标
CORE_METRICS=$(echo "${TEST_OUTPUT}" | grep -E "精确匹配率|平均相似度|总耗时")
if [ -n "$CORE_METRICS" ]; then
    echo "$CORE_METRICS" | while IFS= read -r line; do
        echo "$line" | tee -a "${LOG_FILE}"
    done
fi

# 提取相似度分布（修复匹配模式 - 支持所有区间和百分比格式）
SIMILARITY_DIST=$(echo "${TEST_OUTPUT}" | grep -E "[0-9]\.[0-9]-[0-9]\.[0-9]:[[:space:]]*[0-9]+个样本[[:space:]]*\([0-9]+\.[0-9]+%\)")
if [ -n "$SIMILARITY_DIST" ]; then
    echo "$SIMILARITY_DIST" | while IFS= read -r line; do
        echo "$line" | tee -a "${LOG_FILE}"
    done
else
    # 如果上面的模式没匹配到，尝试更宽松的匹配
    FALLBACK_DIST=$(echo "${TEST_OUTPUT}" | grep -E ":[[:space:]]*[0-9]+个样本")
    if [ -n "$FALLBACK_DIST" ]; then
        echo "$FALLBACK_DIST" | while IFS= read -r line; do
            echo "$line" | tee -a "${LOG_FILE}"
        done
    fi
fi

# 提取难度分析（修复匹配模式 - 包含相似度）
DIFFICULTY_ANALYSIS=$(echo "${TEST_OUTPUT}" | grep -E "Easy:.*样本.*EMR:.*%.*相似度:|Medium:.*样本.*EMR:.*%.*相似度:|Hard:.*样本.*EMR:.*%.*相似度:")
if [ -n "$DIFFICULTY_ANALYSIS" ]; then
    echo "$DIFFICULTY_ANALYSIS" | while IFS= read -r line; do
        echo "$line" | tee -a "${LOG_FILE}"
    done
else
    # 尝试备用匹配模式
    FALLBACK_DIFF=$(echo "${TEST_OUTPUT}" | grep -E "Easy:.*样本|Medium:.*样本|Hard:.*样本")
    if [ -n "$FALLBACK_DIFF" ]; then
        echo "$FALLBACK_DIFF" | while IFS= read -r line; do
            echo "$line" | tee -a "${LOG_FILE}"
        done
    fi
fi

# 检查是否有结果文件信息
RESULT_INFO=$(echo "${TEST_OUTPUT}" | grep -E "结果文件|结果已保存")
if [ -n "$RESULT_INFO" ]; then
    echo "$RESULT_INFO" | while IFS= read -r line; do
        echo "[INFO] $line" | tee -a "${LOG_FILE}"
    done
fi

# 根据退出码显示状态
if [ ${EXIT_CODE} -eq 0 ]; then
    log "[SUCCESS] 实验完成"
else
    log "[ERROR] 测试异常退出，代码: ${EXIT_CODE}"
    # 输出错误信息
    echo "${TEST_OUTPUT}" | grep -E "错误|Error|ERROR|异常|Exception" | tail -5 >> "${LOG_FILE}"
fi

log "[INFO] 测试流程结束"
log "[INFO] 完整日志详见: ${LOG_FILE}"