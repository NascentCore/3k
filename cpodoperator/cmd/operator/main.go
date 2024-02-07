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
	"flag"
	"os"

	// Import all Kubernetes client auth plugins (e.g. Azure, GCP, OIDC, etc.)
	// to ensure that exec-entrypoint and run can make use of them.
	_ "k8s.io/client-go/plugin/pkg/client/auth"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	cpodv1 "github.com/NascentCore/cpodoperator/api/v1"
	cpodv1beta1 "github.com/NascentCore/cpodoperator/api/v1beta1"
	"github.com/NascentCore/cpodoperator/internal/controller"

	kservev1beta1 "github.com/kserve/kserve/pkg/apis/serving/v1beta1"
	mpiv2 "github.com/kubeflow/mpi-operator/pkg/apis/kubeflow/v2beta1"
	tov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	//+kubebuilder:scaffold:imports
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))

	utilruntime.Must(cpodv1beta1.AddToScheme(scheme))
	utilruntime.Must(cpodv1.AddToScheme(scheme))
	utilruntime.Must(tov1.AddToScheme(scheme))
	utilruntime.Must(mpiv2.AddToScheme(scheme))
	utilruntime.Must(kservev1beta1.AddToScheme(scheme))
	//+kubebuilder:scaffold:scheme
}

func main() {
	var metricsAddr string
	var enableLeaderElection bool
	var probeAddr string
	var downloaderImage string
	var tensorrtConvertImage string
	var storageClassName string
	var modelUploadJobImage string
	var modelUploadJobBackoffLimit int
	var modelUploadOssBucketName string
	var inferenceIngressDomain string
	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "The address the metric endpoint binds to.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	flag.BoolVar(&enableLeaderElection, "leader-elect", false,
		"Enable leader election for controller manager. "+
			"Enabling this will ensure there is only one active controller manager.")
	flag.StringVar(&downloaderImage, "artifact-downloader-image", "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/downloader:v1.0.0", "The artifact download job image ")
	flag.StringVar(&tensorrtConvertImage, "tensorrt-convert-image", "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/trtinfer:build-engine", "The image to implement tensorrt convert job")
	flag.StringVar(&storageClassName, "storageClassName", "ceph-filesystem", "which storagecalss the artifact downloader should create")
	flag.StringVar(&modelUploadJobImage, "model-upload-job-image", "sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/modeluploader:ea1b518", "the image of model upload job")
	flag.IntVar(&modelUploadJobBackoffLimit, "model-upload-job-backoff-lmit", 10, "the backoff limit of model upload job")
	flag.StringVar(&modelUploadOssBucketName, "model-upload-job-bucket-name", "sxwl-ai-test", "the oss bucket name of model upload job")
	flag.StringVar(&inferenceIngressDomain, "inference-ingress-domain", "llm.sxwl.ai", "the domain of inference ingress")
	opts := zap.Options{
		Development: true,
	}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:                 scheme,
		Metrics:                metricsserver.Options{BindAddress: metricsAddr},
		HealthProbeBindAddress: probeAddr,
		LeaderElection:         enableLeaderElection,
		LeaderElectionID:       "2ca7307e.cpod",
		// LeaderElectionReleaseOnCancel defines if the leader should step down voluntarily
		// when the Manager ends. This requires the binary to immediately end when the
		// Manager is stopped, otherwise, this setting is unsafe. Setting this significantly
		// speeds up voluntary leader transitions as the new leader don't have to wait
		// LeaseDuration time first.
		//
		// In the default scaffold provided, the program ends immediately after
		// the manager stops, so would be fine to enable this option. However,
		// if you are doing or is intended to do any operation such as perform cleanups
		// after the manager stops then its usage might be unsafe.
		// LeaderElectionReleaseOnCancel: true,
	})
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}

	if err = (&controller.CPodJobReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("cpodjob-controller"),
		Option: &controller.CPodJobOption{
			StorageClassName:           storageClassName,
			ModelUploadJobImage:        modelUploadJobImage,
			ModelUploadJobBackoffLimit: int32(modelUploadJobBackoffLimit),
			ModelUploadOssBucketName:   modelUploadOssBucketName,
		},
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "CPodJob")
		os.Exit(1)
	}

	if err = (&controller.InferenceReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("inference-controller"),
		Options: controller.InferenceOptions{
			Domain: inferenceIngressDomain,
		},
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "Inference")
		os.Exit(1)
	}

	if err = (&controller.ModelStorageReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
		Option: &controller.ModelStorageOption{
			DownloaderImage:      downloaderImage,
			StorageClassName:     storageClassName,
			TensorRTConvertImage: tensorrtConvertImage,
		},
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "ModelStorage")
		os.Exit(1)
	}
	//+kubebuilder:scaffold:builder

	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		os.Exit(1)
	}
	// TODO: @sxwl-donggang 自定Ready checker
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up ready check")
		os.Exit(1)
	}

	ctx := ctrl.SetupSignalHandler()

	setupLog.Info("starting manager")
	if err := mgr.Start(ctx); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}
