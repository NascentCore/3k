package log

import (
	"os"

	"go.uber.org/zap"
)

// NO_TEST_NEEDED

var Logger *zap.Logger = initLogger()
var SLogger *zap.SugaredLogger = Logger.Sugar()

func initLogger() *zap.Logger {
	var logger *zap.Logger
	var err error
	// TODO: 根据环境变量生成不同的Logger
	deploy := os.Getenv("DEPLOY")
	if deploy == "DEBUG" || deploy == "DEV" || deploy == "TEST" {
		logger, err = zap.NewDevelopment()
	} else {
		logger, err = zap.NewProduction()
	}
	if err != nil {
		panic("logger init err")
	}
	return logger
}
