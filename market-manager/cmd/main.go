package main

import (
	"flag"
	"net/http"
	"strings"
	mmInint "sxwl/3k/mm/cmd/init"
	"sxwl/3k/mm/internal/handler"
	"sxwl/3k/mm/internal/router"

	"github.com/gin-gonic/gin"
	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

func main() {
	opts := mmInint.NewMarkeManagerOptions()
	opts.AddFlags(pflag.CommandLine)
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
	pflag.Parse()
	defer glog.Flush()
	warnMsg := opts.VerifyParams()
	if len(warnMsg) > 0 {
		glog.Fatalf(strings.Join(warnMsg, ","))
		return
	}
	glog.V(0).Infof("verify params success the port is %s,%s, the db dsn is %s", opts.Port, opts.DbDsn)
	handler.NewMmDb(opts.DbDsn)
	glog.V(0).Info("start marke manager ...")
	gin.SetMode(gin.DebugMode)
	//gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	router.CpodRouter(r)
	s := &http.Server{
		Addr:           opts.Port,
		Handler:        r,
		ReadTimeout:    0,
		WriteTimeout:   0,
		MaxHeaderBytes: 0,
	}
	s.ListenAndServe()
}
