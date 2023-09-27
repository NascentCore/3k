// NO_TEST_NEEDED
package init

import (
	"flag"

	"github.com/spf13/pflag"
)

type MarkeManagerOptions struct {
	Port      string
	DbDsn     string
	AppMarket string
}

func init() {
	pflag.CommandLine.AddGoFlagSet(flag.CommandLine)
}

func NewMarkeManagerOptions() *MarkeManagerOptions {
	return &MarkeManagerOptions{}
}

func (opts *MarkeManagerOptions) AddFlags(fs *pflag.FlagSet) {
	fs.StringVar(&opts.Port, "port", ":10012", "port")
	fs.StringVar(&opts.DbDsn, "dbDsn", "root:root@tcp(127.0.0.1:3306)/test?charset=utf8mb4&parseTime=True&loc=Local", "db dsn")
	fs.StringVar(&opts.AppMarket, "appMarket", "sxwl.core.ai", "app market url")
}

func (opts *MarkeManagerOptions) VerifyParams() []string {
	var errs []string
	if opts.Port == "" {
		errs = append(errs, "port is required")
	}
	if opts.DbDsn == "" {
		errs = append(errs, "db-dsn is required")
	}
	if opts.AppMarket == "" {
		errs = append(errs, "app-market is required")
	}
	return errs
}
