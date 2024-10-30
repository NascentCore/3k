# Makefile

# 定义脚本文件名
SCRIPT = make-install.sh

# 定义默认目标
.PHONY: operators
operators: operators

# 定义 operators 目标，执行脚本
.PHONY: operators
operators: $(SCRIPT)
	@echo "Executing online install script..."
	@sh $(SCRIPT)

# 确保脚本文件存在
$(SCRIPT):
	@echo "Error: $(SCRIPT) not found!"
	@exit 1

# 允许用户清理生成的文件（如果需要）
.PHONY: clean
clean:
	@echo "Cleaning up..."