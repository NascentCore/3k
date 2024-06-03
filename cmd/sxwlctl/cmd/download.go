/*
Copyright © 2024 NAME HERE <EMAIL ADDRESS>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package cmd

import (
	"fmt"
	"log"
	"path"
	"sxwl/3k/cmd/sxwlctl/internal/auth"
	"sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/fs"
	"sxwl/3k/pkg/storage"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	name   string
	outDir string
)

// downloadCmd represents the download command
var downloadCmd = &cobra.Command{
	Use:   "download",
	Short: "",
	Long:  ``,
	Run: func(cmd *cobra.Command, args []string) {
		err := fs.MakeDir(outDir)
		if err != nil {
			log.Fatal(err.Error())
		}

		switch typ {
		case consts.Model, consts.Dataset, consts.Adapter:
		default:
			log.Fatal("data_type should be [model|dataset|adapter]")
		}

		token := viper.GetString("token")
		if token == "" {
			log.Fatal("Please set token in ~/.sxwlctl.yaml")
		}
		accessID, accessKey, _, err := auth.GetAccessByToken(token)
		if err != nil {
			log.Fatal("Please check token and auth_url in config file")
		}

		endpoint := viper.GetString("endpoint")
		if endpoint == "" {
			log.Fatal("Please set endpoint in ~/.sxwlctl.yaml")
		}

		bucket := viper.GetString("bucket")
		if bucket == "" {
			log.Fatal("Please set bucket in ~/.sxwlctl.yaml")
		}

		conf := Config{
			Endpoint:  endpoint,
			AccessID:  accessID,
			AccessKey: accessKey,
			Bucket:    bucket,
		}

		// init oss client
		storage.InitClient(accessID, accessKey)

		start := time.Now()
		ossPath := ""
		switch typ {
		case consts.Model:
			ossPath = fmt.Sprintf(consts.OSSUserModelPath, name)
		case consts.Dataset:
			ossPath = fmt.Sprintf(consts.OSSUserDatasetPath, name)
		case consts.Adapter:
			ossPath = fmt.Sprintf(consts.OSSUserAdapterPath, name)
		}
		size, err := storage.DownloadDir(conf.Bucket, path.Join(outDir, name), ossPath, verbose)
		if err != nil {
			log.Fatalf("download err: %s", err)
		}
		fmt.Printf("%s has been downloaded. size: %s used time: %s\n", name, fs.FormatBytes(size), time.Since(start).String())
	},
}

func init() {
	rootCmd.AddCommand(downloadCmd)
	downloadCmd.Flags().SortFlags = false
	downloadCmd.Flags().StringVarP(&typ, "type", "t", "model", "[model|dataset|adapter]")
	downloadCmd.Flags().StringVarP(&name, "name", "n", "model", "资源名称，可从页面上复制")
	downloadCmd.Flags().StringVarP(&outDir, "out_dir", "o", "./", "输出目录")
	downloadCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "show verbose logs")
}
