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
	flag.IntVar(&syncPeriod, "sync-period", 10, "the period of every run of synchronizer, unit is second")
	flag.Parse()
	sxwlBaseUrl := os.Getenv("API_ADDRESS") //from configmap provided by cairong
	accessKey := os.Getenv("ACCESS_KEY")    //from configmap provided by cairong
	cpodId := os.Getenv("CPOD_ID")          //from configmap provided by cairong

	cli, err := client.New(clientconfig.GetClientConfig(), client.Options{Scheme: scheme})
	if err != nil {
		panic(err)
	}
	ctx := context.TODO()
	syncManager := synchronizer.NewManager(cpodId, cli, sxwl.NewScheduler(sxwlBaseUrl, accessKey, cpodId),
		time.Duration(syncPeriod)*time.Second, zapr.NewLogger(zap.NewRaw()))
	syncManager.Start(ctx)
	<-ctx.Done()
}
