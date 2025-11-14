#!/bin/bash
# Qwen2-8B记忆率测试脚本

# ==================== 配置区域 ====================
BASE_DIR="/zhangguangyi01/Lianghongjian"
SCRIPT="${BASE_DIR}/code/qwen_test.py"
LOG_DIR="${BASE_DIR}/logs"
RESULT_DIR="${BASE_DIR}/result"
PYTHON_EXEC="/usr/bin/python3.12"
TIMEOUT_DURATION="4h"
MODEL_NAME="Qwen/Qwen2-8B-Instruct"
# ==================================================

# 创建日志文件
LOG_FILE="${LOG_DIR}/qwen_test_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}"

# ==================== 日志函数 ====================
log() {
    echo "$1" | tee -a "${LOG_FILE}"
}

# ==================== 主执行流程 ====================

log "[INFO] 开始Qwen2-8B记忆率测试"
log "[INFO] 启动测试执行"

# 执行测试并捕获输出
TEST_OUTPUT=$(timeout ${TIMEOUT_DURATION} ${PYTHON_EXEC} "${SCRIPT}" 2>&1)
EXIT_CODE=$?

# 记录完整的Python输出到日志文件（不重复显示在控制台）
echo "=== 完整Python输出 ===" >> "${LOG_FILE}"
echo "${TEST_OUTPUT}" >> "${LOG_FILE}"
echo "=== 完整输出结束 ===" >> "${LOG_FILE}"

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