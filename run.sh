BASE_DIR="/zhangguangyi01/Lianghongjian"
SCRIPT="${BASE_DIR}/code/04_baseline_memorization.py"
LOG_DIR="${BASE_DIR}/logs"
LOG_FILE="${LOG_DIR}/experiment_$(date +%Y%m%d_%H%M%S).log"

# 使用系统Python的绝对路径
PYTHON_EXEC="/usr/bin/python3.12"

echo "=== 开始运行记忆率测试 ===" | tee "${LOG_FILE}"
echo "时间: $(date)" | tee -a "${LOG_FILE}"
echo "Python路径: ${PYTHON_EXEC}" | tee -a "${LOG_FILE}"

# 验证Python可执行
if ! [ -x "${PYTHON_EXEC}" ]; then
    echo "❌ 错误：Python不可执行 ${PYTHON_EXEC}" | tee -a "${LOG_FILE}"
    exit 1
fi

# 使用绝对路径运行
timeout 2h "${PYTHON_EXEC}" "${SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
echo "=== 运行结束 ===" | tee -a "${LOG_FILE}"
echo "退出代码: ${EXIT_CODE}" | tee -a "${LOG_FILE}"

[ ${EXIT_CODE} -eq 0 ] && echo "✅ 实验成功完成" || echo "❌ 实验异常退出"
