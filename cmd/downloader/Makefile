.PHONY: build test clean

APP_NAME := app

build:
	@echo "Building $(APP_NAME)..."
	@go build -o $(APP_NAME) main.go

test:
	@echo "Running tests..."
	@go test -v ./...

clean:
	@echo "Cleaning up..."
	@rm -rf $(APP_NAME)