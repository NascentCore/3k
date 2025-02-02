package synchronizer

import (
	"context"
	"time"

	"github.com/NascentCore/cpodoperator/api/v1beta1"
	"github.com/NascentCore/cpodoperator/pkg/provider/litellm"
	"github.com/NascentCore/cpodoperator/pkg/provider/sxwl"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type Runnable interface {
	Start(ctx context.Context)
}

// 管理 portal synchronizer 中的三种计算任务：同步任务、CPod 状态信息上传、CPod 状态收集
type Manager struct {
	// runnables is a collection of all the runnables that are managed by this manager.
	runables []Runnable

	logger logr.Logger

	// 执行的周期，或者说多久进行一次动作
	period time.Duration
}

func NewManager(cpodId, inferImage, embeddingImage, storageClassName string, playgroundBaseURL, accessKey, sxwlInferenceBaseURL, playgroundNamespace string, uploadTrainedModel, autoDownloadResource bool, kubeClient client.Client, scheduler sxwl.Scheduler, period time.Duration, logger logr.Logger) *Manager {
	ch := make(chan sxwl.HeartBeatPayload, 1)
	syncJob := NewSyncJob(kubeClient, scheduler, logger.WithName("syncjob"), uploadTrainedModel, autoDownloadResource, inferImage, embeddingImage, storageClassName)
	uploader := NewUploader(ch, scheduler, period, logger.WithName("uploader"))
	cpodObserver := NewCPodObserver(kubeClient, cpodId, v1beta1.CPOD_NAMESPACE, ch, logger.WithName("cpodobserver"))
	playground := NewPlayground(kubeClient, litellm.NewLitellm(playgroundBaseURL, accessKey), sxwlInferenceBaseURL, playgroundNamespace, logger.WithName("playground"))

	return &Manager{
		runables: []Runnable{
			syncJob,
			uploader,
			cpodObserver,
			playground,
		},
		period: period,
	}
}

func (m *Manager) Start(ctx context.Context) error {
	for _, runable := range m.runables {
		runable := runable
		go wait.UntilWithContext(ctx, func(ctx context.Context) {
			runable.Start(ctx)
		}, m.period)
	}
	return nil
}
