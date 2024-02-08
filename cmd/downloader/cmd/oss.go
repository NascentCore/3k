/*
Copyright © 2024 NAME HERE <EMAIL ADDRESS>

*/
package cmd

import (
	"errors"
	"fmt"
	"os"
	"sxwl/3k/cmd/downloader/internal/consts"
	"sxwl/3k/cmd/downloader/internal/download"
	e "sxwl/3k/cmd/downloader/internal/errors"
	"sxwl/3k/cmd/downloader/internal/oss"
	"sxwl/3k/cmd/downloader/internal/record"
	consts2 "sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/log"

	"github.com/spf13/cobra"
)

// ossCmd represents the oss command
var ossCmd = &cobra.Command{
	Use:   "oss",
	Short: "download oss file",
	Long: fmt.Sprintf(`For example:

downloader oss oss://sxwl-cache/models/nlp_gpt3_text-generation_1.3B \
  -o /data \
  -r crd \
  -g "%s" \
  -v v1 \
  -p modelstorages \
  -n cpod \
  --name model-storage-b93697a593f9bc93
  --endpoint https://oss-cn-beijing.aliyuncs.com
  --access_id xxxxxxxxxx
  --access_key kkkkkkkkkk
`, consts2.ApiGroup),
	Run: func(cmd *cobra.Command, args []string) {
		if len(args) != 1 {
			fmt.Println("please input the oss url")
			os.Exit(1)
		}
		ossUrl := args[0]
		bucket, object, err := oss.ExtractURL(ossUrl)
		if err != nil {
			log.SLogger.Infof("oss url %s is error", ossUrl)
			os.Exit(1)
		}
		c.Bucket = bucket
		c.Object = object

		if c.Record == consts.CRD {
			c.IsCRD = true
		} else {
			c.IsCRD = false
		}

		log.SLogger.Infof("downloader args : %+v", c)

		downloader := download.NewDownloader(consts.OSSDownloader, c)
		recorder := record.NewRecorder(c.Record, c.RecordConfig)

		err = download.Download(c, downloader, recorder)
		if err != nil {
			if errors.Is(err, e.ErrJobComplete) {
				log.SLogger.Infof("download %s has completed", ossUrl)
				os.Exit(0)
			}
			if errors.Is(err, e.ErrJobDownloading) {
				log.SLogger.Infof("download %s is downloading", ossUrl)
				os.Exit(0)
			}

			log.SLogger.Errorf("download err:%s", err)
			os.Exit(1)
		}
	},
}

func init() {
	rootCmd.AddCommand(ossCmd)
	ossCmd.Flags().SortFlags = false
	ossCmd.Flags().StringVarP(&c.OutDir, "output_dir", "o", "/data", "output dir")
	ossCmd.Flags().Int64VarP(&c.Total, "total_size", "t", 0, "total size of the repo")
	ossCmd.Flags().StringVar(&c.Endpoint, "endpoint", "https://oss-cn-beijing.aliyuncs.com", "oss endpoint")
	ossCmd.Flags().StringVar(&c.AccessID, "access_id", "", "oss accessId")
	ossCmd.Flags().StringVar(&c.AccessKey, "access_key", "", "oss accessKey")
	ossCmd.Flags().StringVarP(&c.Record, "record", "r", "crd", "record type (crd)")
	ossCmd.Flags().StringVarP(&c.Group, "group", "g", consts2.ApiGroup, "CRD group")
	ossCmd.Flags().StringVarP(&c.Version, "version", "v", "v1", "CRD version")
	ossCmd.Flags().StringVarP(&c.Plural, "plural", "p", "modelstorages", "CRD resource")
	ossCmd.Flags().StringVarP(&c.Namespace, "namespace", "n", "cpod", "CRD namespace")
	ossCmd.Flags().StringVar(&c.Name, "name", "", "CRD name")
}
