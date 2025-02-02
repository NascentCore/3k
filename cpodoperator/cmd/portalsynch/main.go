/*
Copyright 2023.

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

package main

import (
	"context"
	"flag"
	"os"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	_ "k8s.io/client-go/plugin/pkg/client/auth"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"

	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	cpodv1 "github.com/NascentCore/cpodoperator/api/v1"
	cpodv1beta1 "github.com/NascentCore/cpodoperator/api/v1beta1"
	clientconfig "github.com/NascentCore/cpodoperator/cmd/portalsynch/client-config"
	"github.com/NascentCore/cpodoperator/internal/synchronizer"
	"github.com/NascentCore/cpodoperator/pkg/provider/sxwl"

	"github.com/go-logr/zapr"
)

var scheme = runtime.NewScheme()

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(cpodv1beta1.AddToScheme(scheme))
	utilruntime.Must(cpodv1.AddToScheme(scheme))
}

func main() {
	var syncPeriod int
	var uploadTrainedModel bool
	var autoDownloadResource bool
	var inferImage, embeddingImage string
	var storageClassName string
	var LitellmURL, LitellmAccessKey, LitellmNamespace string
	var sxwlInferenceBaseURL string
	flag.IntVar(&syncPeriod, "sync-period", 10, "the period of every run of synchronizer, unit is second")
	flag.BoolVar(&uploadTrainedModel, "upload-trained-model", true, "whether to upload trained model to sxwl or not")
	flag.BoolVar(&autoDownloadResource, "auto-download-resource", false, "whether download model or dataset when not exists")
	flag.StringVar(&inferImage, "infer-image", "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/llamafactory:inference-v1", "the image for inference")
	flag.StringVar(&embeddingImage, "embedding-image", "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/infinity:latest", "the image for embedding")
	flag.StringVar(&storageClassName, "storageClassName", "juicefs-sc", "which storagecalss the artifact downloader should create")
	flag.StringVar(&LitellmURL, "litellm-url", "http://playground.llm.sxwl.ai:30005", "the url for litellm")
	flag.StringVar(&LitellmAccessKey, "litellm-access-key", "sk-1234", "the access key for litellm")
	flag.StringVar(&LitellmNamespace, "litellm-namespace", "user-7e687ea0-844b-42c3-9005-ec9ddf4ae863", "the namespace for litellm")
	flag.StringVar(&sxwlInferenceBaseURL, "sxwl-inference-base-url", "http://master.llm.sxwl.ai:30005", "the base url for sxwl inference")
	flag.Parse()
	sxwlBaseUrl := os.Getenv("API_ADDRESS") // from configmap provided by cairong
	accessKey := os.Getenv("ACCESS_KEY")    // from configmap provided by cairong
	cpodId := os.Getenv("CPOD_ID")          // from configmap provided by cairong

	cli, err := client.New(clientconfig.GetClientConfig(), client.Options{Scheme: scheme})
	if err != nil {
		panic(err)
	}
	ctx := context.TODO()
	syncManager := synchronizer.NewManager(cpodId, inferImage, embeddingImage, storageClassName, LitellmURL, LitellmAccessKey, sxwlInferenceBaseURL, LitellmNamespace, uploadTrainedModel, autoDownloadResource, cli,
		sxwl.NewScheduler(sxwlBaseUrl, accessKey, cpodId),
		time.Duration(syncPeriod)*time.Second, zapr.NewLogger(zap.NewRaw()))
	syncManager.Start(ctx)
	<-ctx.Done()
}
