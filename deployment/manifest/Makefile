# Makefile

# 定义脚本文件名
SCRIPT = make-package.sh

# 定义默认目标
.PHONY: all
all: generate

# 定义 generate 目标，执行脚本
.PHONY: generate
generate: $(SCRIPT)
	@echo "Executing generate script..."
	@sh $(SCRIPT)

# 确保脚本文件存在
$(SCRIPT):
	@echo "Error: $(SCRIPT) not found!"
	@exit 1

# 允许用户清理生成的文件（如果需要）
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf 3k-artifacts.tar.gz