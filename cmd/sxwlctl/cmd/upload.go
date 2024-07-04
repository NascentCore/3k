package cmd

import (
	"fmt"
	"log"
	"os"
	"path"
	"path/filepath"
	"sxwl/3k/cmd/sxwlctl/internal/auth"
	consts2 "sxwl/3k/cmd/sxwlctl/internal/consts"
	"sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/fs"
	"sxwl/3k/pkg/storage"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	dir      string
	resource string
	template string
	public   bool
	owner    string
	verbose  bool
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
	Long:  `上传本地模型、数据集、适配器`,
	Run: func(cmd *cobra.Command, args []string) {
		// check dir
		if !fs.IsDirExist(dir) {
			fmt.Println("Please input a correct local dir")
			os.Exit(1)
		}

		token := viper.GetString("token")
		if token == "" {
			fmt.Println("Please input a sxwl token")
			os.Exit(1)
		}
		accessID, accessKey, userID, isAdmin, err := auth.GetAccessByToken(token)
		if err != nil {
			fmt.Println("Please check token and auth_url in config file")
			os.Exit(1)
		}

		if public {
			// only admin could upload public resource
			if !isAdmin {
				fmt.Println("Only admin could upload public resource")
				os.Exit(1)
			}

			if owner == "" {
				fmt.Println("Please set owner when uploading public resource")
				os.Exit(1)
			}
		}

		switch resource {
		case consts.Model:
			if template == "" {
				log.Fatalf("Please use -t to set the inference template used for this model")
			}
			// meta file
			err := fs.TouchFile(path.Join(dir, consts2.FileCanFinetune))
			if err != nil {
				log.Fatalf("touch file %s err: %s", consts2.FileCanFinetune, err)
			}

			err = fs.TouchFile(path.Join(dir, consts2.FileCanInference))
			if err != nil {
				log.Fatalf("touch file %s err: %s", consts2.FileCanInference, err)
			}

			err = fs.TouchFile(path.Join(dir, fmt.Sprintf(consts2.FileInferTemplate, template)))
			if err != nil {
				log.Fatalf("touch file %s err: %s", fmt.Sprintf(consts2.FileInferTemplate, template), err)
			}
		case consts.Dataset, consts.Adapter:
		default:
			fmt.Println("data_type should be [model|dataset|adapter]")
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
		prefixFormat := ""
		switch resource {
		case consts.Model:
			if public {
				prefixFormat = consts.OSSPublicModelPath
			} else {
				prefixFormat = consts.OSSUserModelPath
			}
		case consts.Dataset:
			if public {
				prefixFormat = consts.OSSPublicDatasetPath
			} else {
				prefixFormat = consts.OSSUserDatasetPath
			}
		case consts.Adapter:
			if public {
				prefixFormat = consts.OSSPublicAdapterPath
			} else {
				prefixFormat = consts.OSSUserAdapterPath
			}
		}
		if !public {
			owner = userID
		}
		prefix := fmt.Sprintf(prefixFormat, owner)

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
	uploadCmd.Flags().StringVarP(&resource, "resource", "r", "model", "[model|dataset|adapter]")
	uploadCmd.Flags().StringVarP(&dir, "dir", "d", "", "上传的本地文件夹路径")
	uploadCmd.Flags().StringVarP(&template, "template", "t", "", "模型推理使用的template")
	uploadCmd.Flags().BoolVar(&public, "public", false, "上传至公共空间")
	uploadCmd.Flags().StringVar(&owner, "owner", "", "公共资源的所有者，仅在--public时需要")
	uploadCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "show verbose logs")
}
