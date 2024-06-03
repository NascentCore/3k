package cmd

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sxwl/3k/cmd/sxwlctl/internal/auth"
	"sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/fs"
	"sxwl/3k/pkg/storage"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	dir     string
	typ     string
	verbose bool
)

type Config struct {
	Endpoint  string
	AccessID  string
	AccessKey string
	Bucket    string
}

// uploadCmd represents the upload command
var uploadCmd = &cobra.Command{
	Use:   "upload",
	Short: "upload",
	Long:  ``,
	Run: func(cmd *cobra.Command, args []string) {
		if !fs.IsDirExist(dir) {
			fmt.Println("Please input a correct local dir")
			os.Exit(1)
		}

		switch typ {
		case consts.Model, consts.Dataset, consts.Adapter:
		default:
			fmt.Println("data_type should be [model|dataset|adapter]")
			os.Exit(1)
		}

		token := viper.GetString("token")
		if token == "" {
			fmt.Println("Please input a sxwl token")
			os.Exit(1)
		}
		accessID, accessKey, userID, err := auth.GetAccessByToken(token)
		if err != nil {
			fmt.Println("Please check token and auth_url in config file")
			os.Exit(1)
		}

		conf := Config{
			Endpoint:  "https://oss-cn-beijing.aliyuncs.com",
			AccessID:  accessID,
			AccessKey: accessKey,
			Bucket:    "sxwl-cache",
		}

		// init oss client
		storage.InitClient(accessID, accessKey)

		start := time.Now()
		prefix := ""
		switch typ {
		case consts.Model:
			prefix = fmt.Sprintf(consts.OSSUserModelPath, userID)
		case consts.Dataset:
			prefix = fmt.Sprintf(consts.OSSUserDatasetPath, userID)
		case consts.Adapter:
			prefix = fmt.Sprintf(consts.OSSUserAdapterPath, userID)
		}
		// size, err := upload(conf, path.Join(prefix, filepath.Base(dir)), dir)
		size, err := storage.UploadDir(conf.Bucket, dir, path.Join(prefix, filepath.Base(dir)), verbose)
		if err != nil {
			fmt.Printf("upload err: %s\n", err)
			os.Exit(1)
		} else {
			fmt.Printf("%s has been uploaded. size: %s used time: %s\n", dir, fs.FormatBytes(size), time.Since(start).String())
		}
	},
}

func init() {
	rootCmd.AddCommand(uploadCmd)
	uploadCmd.Flags().SortFlags = false
	uploadCmd.Flags().StringVarP(&typ, "type", "t", "model", "[model|dataset|adapter]")
	uploadCmd.Flags().StringVarP(&dir, "dir", "d", "", "上传的本地文件夹路径")
	uploadCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "show verbose logs")
	viper.SetDefault("auth_url", "https://llm.sxwl.ai/api/uploader_access")
}
