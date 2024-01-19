/*
Copyright © 2023 NAME HERE <EMAIL ADDRESS>
*/
package cmd

import (
	"errors"
	"fmt"
	"os"
	"sxwl/3k/cmd/downloader/internal/consts"
	"sxwl/3k/cmd/downloader/internal/download"
	e "sxwl/3k/cmd/downloader/internal/errors"
	consts2 "sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/log"

	"github.com/spf13/cobra"
)

var (
	c download.Config
)

// gitCmd represents the git command
var gitCmd = &cobra.Command{
	Use:   "git url",
	Short: "download data from git url",
	Long: fmt.Sprintf(`download data from git url. Example:
    downloader git https://www.modelscope.cn/Cherrytest/rot_bgr.git -o /data \
        -r crd \
        -g "%s" \
        -v v1 \
        -p modelstorages \
        -n cpod \
        --name model-storage-b93697a593f9bc93
		--depth 1
`, consts2.ApiGroup),
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) != 1 {
			fmt.Println("please input the git url for downloading")
			os.Exit(1)
		}
		c.GitUrl = args[0]
		if c.Record == consts.CRD {
			c.IsCRD = true
		} else {
			c.IsCRD = false
		}

		log.SLogger.Infof("downloader args : %+v", c)

		err := download.GitDownload(c)
		if err != nil {
			if errors.Is(err, e.ErrJobComplete) {
				log.SLogger.Infof("download %s has completed", c.GitUrl)
				os.Exit(0)
			}
			if errors.Is(err, e.ErrJobDownloading) {
				log.SLogger.Infof("download %s is downloading", c.GitUrl)
				os.Exit(0)
			}

			log.SLogger.Errorf("download err:%s", err)
			os.Exit(1)
		}
	},
}

func init() {
	rootCmd.AddCommand(gitCmd)
	gitCmd.Flags().SortFlags = false
	gitCmd.Flags().StringVarP(&c.OutDir, "output_dir", "o", "/data", "output dir")
	gitCmd.Flags().Int64VarP(&c.Total, "total_size", "t", 0, "total size of the repo")
	gitCmd.Flags().StringVarP(&c.Record, "record", "r", "crd", "record type (crd)")
	gitCmd.Flags().StringVarP(&c.Group, "group", "g", consts2.ApiGroup, "CRD group")
	gitCmd.Flags().StringVarP(&c.Version, "version", "v", "v1", "CRD version")
	gitCmd.Flags().StringVarP(&c.Plural, "plural", "p", "modelstorages", "CRD resource")
	gitCmd.Flags().StringVarP(&c.Namespace, "namespace", "n", "cpod", "CRD namespace")
	gitCmd.Flags().StringVar(&c.Name, "name", "", "CRD name")
	gitCmd.Flags().UintVarP(&c.Depth, "depth", "d", 0,
		"--depth likes git, default is full history")
}
