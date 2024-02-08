package config

import (
	oss2 "sxwl/3k/cmd/downloader/internal/oss"
	"sxwl/3k/cmd/downloader/internal/record"
)

type Config struct {
	record.RecordConfig
	oss2.OSSConfig
	GitUrl string
	Total  int64
	OutDir string
	Record string
	IsCRD  bool
}
