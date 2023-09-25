package handler

import (
	"sync"

	"github.com/golang/glog"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
)

var (
	DbClinet *gorm.DB
	dbOnce   sync.Once
)

func NewMmDb(dsn string) {
	if dsn == "" {
		glog.Error("db dsn can not be empty")
		return
	}
	dbOnce.Do(func() {
		var err error
		DbClinet, err = gorm.Open(mysql.Open(dsn), &gorm.Config{})
		if err != nil {
			glog.Error("connect to the mysql error, please check the connection !!! ", err)
			//panic(err)
		}
	})
}
