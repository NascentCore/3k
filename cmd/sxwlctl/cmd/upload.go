package cmd

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path"
	"path/filepath"
	"sxwl/3k/cmd/sxwlctl/internal/sxy"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/fs"
	"sxwl/3k/pkg/storage"
	"time"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	dir            string
	resource       string
	template       string
	baseModel      string
	category       string
	public         bool
	owner          string
	verbose        bool
	datasetPreview string
	datasetTotal   int64
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
		accessID, accessKey, userID, isAdmin, err := sxy.GetAccessByToken(token)
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
			if category != consts.ModelCategoryChat && category != consts.ModelCategoryEmbedding {
				log.Fatalf("Please use --category to set the category of this model")
			}
		case consts.Adapter:
			if baseModel == "" {
				log.Fatalf("Please use --base_model to set the base model for the adapter")
			}
		case consts.Dataset:
			datasetPreview, err = fs.PreviewJSONArray(filepath.Join(dir, "dataset.json"), 5)
			if err != nil {
				log.Printf("Please check dataset.json in the dataset dir: %v", err)
			}

			datasetTotal, err = fs.CountJSONArray(filepath.Join(dir, "dataset.json"))
			if err != nil {
				log.Printf("Please check dataset.json in the dataset dir: %v", err)
			}
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
		} else {
			userID = "public"
		}
		prefix := fmt.Sprintf(prefixFormat, owner)

		size, err := storage.UploadDir(conf.Bucket, dir, path.Join(prefix, filepath.Base(dir)), verbose)
		if err != nil {
			fmt.Printf("upload err: %s\n", err)
			os.Exit(1)
		} else {
			resourceName := fmt.Sprintf("%s/%s", owner, filepath.Base(dir))
			resourceID := ""
			metaBytes := []byte("{}")

			switch resource {
			case consts.Model:
				resourceID = storage.ModelCRDName(storage.ResourceToOSSPath(consts.Model, resourceName))
				metaBytes, _ = json.Marshal(model.OssResourceModelMeta{
					Template:     template,
					Category:     category,
					CanFinetune:  true,
					CanInference: true,
				})
			case consts.Dataset:
				resourceID = storage.DatasetCRDName(storage.ResourceToOSSPath(consts.Dataset, resourceName))
				metaBytes, _ = json.Marshal(model.OssResourceDatasetMeta{
					Preview: datasetPreview,
					Total:   datasetTotal,
					Size:    size,
				})
			case consts.Adapter:
				metaBytes, _ = json.Marshal(model.OssResourceAdapterMeta{
					BaseModel: baseModel,
				})
				resourceID = storage.AdapterCRDName(storage.ResourceToOSSPath(consts.Adapter, resourceName))
			}

			meta := string(metaBytes)

			err = sxy.AddResource(token, sxy.Resource{
				ResourceID:   resourceID,
				ResourceType: resource,
				ResourceName: resourceName,
				ResourceSize: size,
				IsPublic:     public,
				UserID:       userID,
				Meta:         meta,
			})
			if err != nil {
				fmt.Printf("add resource err: %s\n", err)
				os.Exit(1)
			}

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
	uploadCmd.Flags().StringVar(&category, "category", "chat", "模型的类型[chat|embedding]")
	uploadCmd.Flags().StringVar(&baseModel, "base_model", "", "适配器的基底模型")
	uploadCmd.Flags().BoolVar(&public, "public", false, "上传至公共空间")
	uploadCmd.Flags().StringVar(&owner, "owner", "", "公共资源的所有者，仅在--public时需要")
	uploadCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "show verbose logs")
}
