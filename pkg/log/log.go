package log

import (
	"sxwl/3k/pkg/config"

	"go.uber.org/zap"
)

// NO_TEST_NEEDED

var Logger *zap.Logger = initLogger()
var SLogger *zap.SugaredLogger = Logger.Sugar()

func initLogger() *zap.Logger {
	var logger *zap.Logger
	var err error
	// TODO: æ ¹æ®ç¯å¢åéçæä¸åçLogger
	deploy := config.DEPLOY
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
