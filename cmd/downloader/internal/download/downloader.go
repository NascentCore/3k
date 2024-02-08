package download

import (
	"fmt"
	"sxwl/3k/cmd/downloader/internal/consts"
	"sxwl/3k/cmd/downloader/internal/download/config"
	"sxwl/3k/cmd/downloader/internal/download/git"
	"sxwl/3k/cmd/downloader/internal/download/oss"
)

type Downloader interface {
	Download() error
}

func NewDownloader(typ string, c config.Config) Downloader {
	switch typ {
	case consts.GitDownloader:
		return git.NewDownloader(c)
	case consts.OSSDownloader:
		return oss.NewDownloader(c)
	default:
		panic(fmt.Sprintf("downloader type: %s not support", typ))
	}
}
