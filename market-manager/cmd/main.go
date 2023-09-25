// NO_TEST_NEEDED
package main

import (
	"flag"
	"net/http"
	"strings"
	_ "sxwl/mm/cmd/docs"
	mmInit "sxwl/mm/cmd/init"
	"sxwl/mm/internal/handler"
	"sxwl/mm/internal/router"

	"github.com/gin-gonic/gin"
	"github.com/golang/glog"
	"github.com/spf13/pflag"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

// @title Market Manager swagger
// @version 1.0
// @description Market Manager API Server
// @BasePath /api/v1/
// @host localhost:10012
func main() {
	opts := mmInit.NewMarkeManagerOptions()
	opts.AddFlags(pflag.CommandLine)
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	pflag.Parse()
	defer glog.Flush()
	warnMsg := opts.VerifyParams()
	if len(warnMsg) > 0 {
		glog.Fatalf(strings.Join(warnMsg, ","))
		return
	}
	glog.V(0).Infof("verify params success the port is %s, the db dsn is %s,the app-market url is %s", opts.Port, opts.DbDsn, opts.AppMarket)
	handler.NewMmDb(opts.DbDsn)
	glog.V(0).Info("start market manager ...")
	gin.SetMode(gin.DebugMode)
	//gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	router.CpodRouter(r)
	r.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))
	s := &http.Server{
		Addr:           opts.Port,
		Handler:        r,
		ReadTimeout:    0,
		WriteTimeout:   0,
		MaxHeaderBytes: 0,
	}
	s.ListenAndServe()
}
