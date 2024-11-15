package controller

import (
	"context"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	cpodv1 "github.com/NascentCore/cpodoperator/api/v1"
	"github.com/NascentCore/cpodoperator/api/v1beta1"
	cpodv1beta1 "github.com/NascentCore/cpodoperator/api/v1beta1"
	kservev1beta1 "github.com/kserve/kserve/pkg/apis/serving/v1beta1"
	rayv1 "github.com/ray-project/kuberay/ray-operator/apis/ray/v1"
	"github.com/sirupsen/logrus"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	netv1 "k8s.io/api/networking/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	"knative.dev/pkg/apis"
	duckv1 "knative.dev/pkg/apis/duck/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type InferenceOptions struct {
	// IngressClass string
	Domain              string
	InferenceWebuiImage string
	InferencePathPrefix string
	OssOption           OssOption
	StorageClassName    string
	RayImage            string
}

// CPodJobReconciler reconciles a CPodJob object
type InferenceReconciler struct {
	client.Client
	Scheme *runtime.Scheme

	// Recorder is an event recorder for recording Event resources to the
	// Kubernetes API.
	Recorder record.EventRecorder

	Options InferenceOptions
}

//+kubebuilder:rbac:groups=cpod.cpod,resources=inferences,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=cpod.cpod,resources=inferences/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=cpod.cpod,resources=inferences/finalizers,verbs=update
//+kubebuilder:rbac:groups=serving.kserve.io,resources=inferenceservices,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=serving.kserve.io,resources=inferenceservices/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=networking.k8s.io,resources=ingresses,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups="apps",resources=deployments,verbs=get;list;watch;create;update;patch;delete

func (i *InferenceReconciler) Reconcile(ctx context.Context, req ctrl.Request) (_ ctrl.Result, reterr error) {
	logger := log.FromContext(ctx)

	inference := &cpodv1beta1.Inference{}
	if err := i.Get(ctx, req.NamespacedName, inference); client.IgnoreNotFound(err) != nil {
		logger.Error(err, "unable to fetch Inference")
		return ctrl.Result{}, err
	}

	if inference.DeletionTimestamp != nil {
		return ctrl.Result{}, nil
	}

	inferenceDeepcopy := inference.DeepCopy()
	defer func() {
		if !equality.Semantic.DeepEqual(inference.Status, inferenceDeepcopy.Status) {
			if err := i.Client.Status().Update(ctx, inferenceDeepcopy); err != nil {
				logger.Error(err, "unable to update CPodJob status")
				reterr = err
			}
		}
	}()

	// prepare model and adapter
	if err := i.prepareData(ctx, inference); err != nil {
		logrus.Error("prepare model and adapter failed", "err", err, "inference", inference)
		inferenceDeepcopy.Status.DataReady = false
		return ctrl.Result{}, err
	}
	inferenceDeepcopy.Status.DataReady = true

	if inference.Spec.Backend == cpodv1beta1.InferenceBackendRay {
		rayService, err := i.prepareRay(ctx, inference)
		if err != nil {
			logger.Error(err, "unable to create ray service")
			return ctrl.Result{}, err
		}
		if err := i.prepareWebUI(ctx, inference); err != nil {
			logger.Error(err, "unable to create web ui")
			return ctrl.Result{}, err
		}
		inferenceDeepcopy.Status.Ready = rayServiceReadiness(rayService.Status)
		if !inferenceDeepcopy.Status.Ready {
			inferenceDeepcopy.Status.Conditions = duckv1.Conditions{
				{
					Type:   "RayServiceReady",
					Status: corev1.ConditionFalse,
					Reason: string(rayService.Status.ServiceStatus),
				},
			}
			return ctrl.Result{RequeueAfter: 20 * time.Second}, nil
		} else {
			inferenceDeepcopy.Status.Conditions = nil
			url := fmt.Sprintf("%v.%v", inference.Name, i.Options.Domain)
			inferenceDeepcopy.Status.URL = &url
		}
	} else {
		inferenceService, err := i.prepareKserve(ctx, inference)
		if err != nil {
			logger.Error(err, "unable to create inferenceService")
			return ctrl.Result{}, err
		}
		err = i.prepareWebUI(ctx, inference)
		if err != nil {
			logger.Error(err, "unable to create web ui")
			return ctrl.Result{}, err
		}
		inferenceDeepcopy.Status.Ready = inferenceServiceReadiness(inferenceService.Status)
		if !inferenceDeepcopy.Status.Ready {
			inferenceDeepcopy.Status.Conditions = inferenceService.Status.Conditions
			return ctrl.Result{RequeueAfter: 20 * time.Second}, nil
		} else {
			inferenceDeepcopy.Status.Conditions = nil
			url := fmt.Sprintf("%v.%v", inference.Name, i.Options.Domain)
			inferenceDeepcopy.Status.URL = &url
		}
	}

	return ctrl.Result{}, nil
}

func (i InferenceReconciler) getInferenceServiceName(inference *cpodv1beta1.Inference) string {
	return inference.Name + "-is"
}

func (i *InferenceReconciler) generateOwnerRefInference(ctx context.Context, inference *cpodv1beta1.Inference) metav1.OwnerReference {
	yes := true
	return metav1.OwnerReference{
		APIVersion:         cpodv1beta1.GroupVersion.String(),
		Kind:               "Inference",
		Name:               inference.Name,
		UID:                inference.GetUID(),
		Controller:         &yes,
		BlockOwnerDeletion: &yes,
	}
}

func (i *InferenceReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&cpodv1beta1.Inference{}).
		Owns(&kservev1beta1.InferenceService{}).
		Complete(i)
}

func (i *InferenceReconciler) prepareIngressOfInferenceService(ctx context.Context, inference *cpodv1beta1.Inference, inferenceservice *kservev1beta1.InferenceService) error {
	ingressName := inference.Name + "-ingress"

	var ingress netv1.Ingress
	if err := i.Client.Get(ctx, types.NamespacedName{Namespace: inference.Namespace, Name: ingressName}, &ingress); err != nil {
		if apierrors.IsNotFound(err) {
			// 获取对应preidector service name
			svcName := PredictorServiceName(inferenceservice.Name)
			// create ingress
			path := filepath.Join(i.Options.InferencePathPrefix, inference.Name+"(/|$)(.*)")
			var rules []netv1.IngressRule
			rules = append(rules, i.generateRule(inference.Name, svcName, path, 80))
			ingress := &netv1.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:      ingressName,
					Namespace: inference.Namespace,
					Annotations: map[string]string{
						"kubernetes.io/ingress.class":                "nginx",
						"nginx.ingress.kubernetes.io/rewrite-target": "/$2",
					},
					OwnerReferences: []metav1.OwnerReference{
						i.generateOwnerRefInference(ctx, inference),
					},
				},
				Spec: netv1.IngressSpec{
					Rules: rules,
				},
			}
			if err := i.Client.Create(ctx, ingress); err != nil {
				ctrl.Log.Info("create ingress failed", "err", err)
				i.Recorder.Eventf(inference, corev1.EventTypeWarning, "CreateIngressFailed", "create ingress failed")
				return err
			}
			return nil
		}
		return err
	}

	return nil
}

func (i *InferenceReconciler) prepareIngressOfRayService(ctx context.Context, inference *cpodv1beta1.Inference, rayService *rayv1.RayService) error {
	ingressName := inference.Name + "-ingress"

	var ingress netv1.Ingress
	if err := i.Client.Get(ctx, types.NamespacedName{Namespace: inference.Namespace, Name: ingressName}, &ingress); err != nil {
		if apierrors.IsNotFound(err) {
			// 获取对应preidector service name
			svcName := rayService.Name + "-serve-svc"
			// 如果 svcName 的长度大于 50，则截取后 50 个字符
			if len(svcName) > 50 {
				svcName = svcName[len(svcName)-50:]
			}

			// create ingress
			path := filepath.Join(i.Options.InferencePathPrefix, inference.Name+"(/|$)(.*)")
			var rules []netv1.IngressRule
			rules = append(rules, i.generateRule(inference.Name, svcName, path, 8000))
			ingress := &netv1.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:      ingressName,
					Namespace: inference.Namespace,
					Annotations: map[string]string{
						"kubernetes.io/ingress.class":                "nginx",
						"nginx.ingress.kubernetes.io/rewrite-target": "/$2",
					},
					OwnerReferences: []metav1.OwnerReference{
						i.generateOwnerRefInference(ctx, inference),
					},
				},
				Spec: netv1.IngressSpec{
					Rules: rules,
				},
			}
			if err := i.Client.Create(ctx, ingress); err != nil {
				ctrl.Log.Info("create ingress failed", "err", err)
				i.Recorder.Eventf(inference, corev1.EventTypeWarning, "CreateIngressFailed", "create ingress failed")
				return err
			}
			return nil
		}
		return err
	}

	return nil
}

func PredictorServiceName(name string) string {
	return name + "-predictor"
}

// func DefaultPredictorServiceName(name string) string {
// 	return name + "-predictor-" + InferenceServiceDefault
// }

func (i *InferenceReconciler) generateRule(inferenceName, componentName string, path string, port int32) netv1.IngressRule {
	pathType := netv1.PathTypeImplementationSpecific
	rule := netv1.IngressRule{
		IngressRuleValue: netv1.IngressRuleValue{
			HTTP: &netv1.HTTPIngressRuleValue{
				Paths: []netv1.HTTPIngressPath{
					{
						Path:     path,
						PathType: &pathType,
						Backend: netv1.IngressBackend{
							Service: &netv1.IngressServiceBackend{
								Name: componentName,
								Port: netv1.ServiceBackendPort{
									Number: port,
								},
							},
						},
					},
				},
			},
		},
	}
	return rule
}

func inferenceServiceReadiness(status kservev1beta1.InferenceServiceStatus) bool {
	return status.Conditions != nil &&
		status.GetCondition(apis.ConditionReady) != nil &&
		status.GetCondition(apis.ConditionReady).Status == corev1.ConditionTrue
}

func rayServiceReadiness(status rayv1.RayServiceStatuses) bool {
	return status.ServiceStatus == rayv1.Running
}

func inferenceServiceURL(status kservev1beta1.InferenceServiceStatus) *string {
	if com, ok := status.Components[kservev1beta1.PredictorComponent]; ok {
		if com.URL != nil {
			URL := com.URL.String()
			return &URL
		}
	}
	return nil
}

func parseModelStorageURI(srcURI string) (modelStorageName string, err error) {
	parts := strings.Split(strings.TrimPrefix(srcURI, cpodv1beta1.ModelStoragePrefix), "/")
	if len(parts) != 1 {
		return "", fmt.Errorf("Invalid URI must be modelstorage://<modelstorage-name>: %s", srcURI)
	}
	return parts[0], nil
}

func (i *InferenceReconciler) prepareData(ctx context.Context, inference *cpodv1beta1.Inference) error {
	modelstorageName := inference.Spec.ModelID
	wg := sync.WaitGroup{}
	errChan := make(chan error, 2)

	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := i.prepareModel(ctx, modelstorageName, inference); err != nil {
			errChan <- err
			return
		}
	}()

	if adapterID, ok := inference.Annotations[cpodv1beta1.CPodAdapterIDAnno]; ok {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := i.prepareAdapter(ctx, adapterID, inference); err != nil {
				errChan <- err
				return
			}
		}()
	}

	wg.Wait()
	if len(errChan) > 0 {
		for err := range errChan {
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (i *InferenceReconciler) prepareAdapter(ctx context.Context, adapterID string, inference *cpodv1beta1.Inference) error {
	logrus.Info("DEBUG prepare adapter", "inference", inference.Spec, "Annotations", inference.Annotations)
	modelSize := int64(0)
	modelReadableName := ""
	modelTemplate := ""
	modelstorageName := adapterID
	isPublic := false
	if inference.Annotations != nil {
		if sizeStr, ok := inference.Annotations[cpodv1beta1.CPodAdapterSizeAnno]; ok {
			size, err := strconv.ParseInt(sizeStr, 10, 64)
			if err != nil {
				return fmt.Errorf("failed to parse adapter size %v: %v", sizeStr, err)
			}
			modelSize = size
		}
		if name, ok := inference.Annotations[v1beta1.CPodAdapterReadableNameAnno]; ok {
			modelReadableName = name
		}

		if metav1.HasAnnotation(inference.ObjectMeta, cpodv1beta1.CPodAdapterIsPublic) {
			isPublic = true
		}

	}
	if isPublic {
		modelName := modelstorageName + v1beta1.CPodPublicStorageSuffix
		ms := &cpodv1.ModelStorage{}
		if err := i.Client.Get(ctx, client.ObjectKey{Namespace: inference.Namespace, Name: modelName}, ms); err != nil {
			if apierrors.IsNotFound(err) {
				publicMs := &cpodv1.ModelStorage{}
				if err := i.Client.Get(ctx, client.ObjectKey{Namespace: v1beta1.CPodPublicNamespace, Name: modelstorageName}, publicMs); err != nil {
					if apierrors.IsNotFound(err) {
						if createdMs, err := createModelstorage(ctx, i.Client, modelstorageName, modelReadableName, modelSize, modelTemplate, v1beta1.CPodPublicNamespace, i.Options.StorageClassName); err != nil {
							return fmt.Errorf("failed to create adapter storage for adapter model %s: %v", modelstorageName, err)
						} else {
							publicMs = createdMs
						}
					} else {
						return fmt.Errorf("failed to get public model %s: %v", modelstorageName, err)
					}
				}
				if publicMs != nil && publicMs.Status.Phase != "done" {
					jobName := "model-" + modelstorageName
					job := &batchv1.Job{}
					if err := i.Client.Get(ctx, client.ObjectKey{Namespace: v1beta1.CPodPublicNamespace, Name: jobName}, job); err != nil {
						if apierrors.IsNotFound(err) {
							if err := CreateDownloadJob(ctx, i.Client, i.Options.OssOption, "model", modelstorageName, modelReadableName, modelSize, inference.Namespace, v1beta1.CPodPublicNamespace); err != nil {
								return fmt.Errorf("failed to create download job for public model %s: %v", modelstorageName, err)
							}
						} else {
							return fmt.Errorf("failed to get public model %s: %v", modelstorageName, err)
						}
					}
					if job.Status.Succeeded != 1 {
						return fmt.Errorf("public model downloader job %s is running: %v", jobName, job.Status.Succeeded)
					}
					return fmt.Errorf("public model %s is not done", modelstorageName)
				}
				if err := CopyPublicModelStorage(ctx, i.Client, modelstorageName, inference.Namespace); err != nil {
					return fmt.Errorf("failed to copy public model %s: %v", modelstorageName, err)
				}
				return nil
			} else {
				return fmt.Errorf("failed to get public model %v's copy  %s: %v", modelstorageName, modelName, err)
			}
		}
		if ms.Status.Phase != "done" {
			return fmt.Errorf("public model copy  %s is not done", modelstorageName)
		}
		return nil
	}
	ms := &cpodv1.ModelStorage{}
	if err := i.Client.Get(ctx, client.ObjectKey{Namespace: inference.Namespace, Name: modelstorageName}, ms); err != nil {
		if apierrors.IsNotFound(err) {
			if createdMs, err := createModelstorage(ctx, i.Client, modelstorageName, modelReadableName, modelSize, modelTemplate, inference.Namespace, i.Options.StorageClassName); err != nil {
				return fmt.Errorf("failed to create model storage for private model %s: %v", modelstorageName, err)
			} else {
				ms = createdMs
			}
		} else {
			return fmt.Errorf("failed to get private model %s: %v", modelstorageName, err)
		}
	}
	if ms != nil && ms.Status.Phase != "done" {
		jobName := "model-" + modelstorageName
		job := &batchv1.Job{}
		if err := i.Client.Get(ctx, client.ObjectKey{Namespace: inference.Namespace, Name: jobName}, job); err != nil {
			if apierrors.IsNotFound(err) {
				if err := CreateDownloadJob(ctx, i.Client, i.Options.OssOption, "model", modelstorageName, modelReadableName, modelSize, inference.Namespace, inference.Namespace); err != nil {
					return fmt.Errorf("failed to create download job for private model %s: %v", modelstorageName, err)
				}
			} else {
				return fmt.Errorf("failed to get private model %s: %v", modelstorageName, err)
			}
		}
		if job.Status.Succeeded != 1 {
			return fmt.Errorf("model downloader job %s is running: %v", jobName, job.Status.Succeeded)
		}
		return fmt.Errorf("private model %s is not done", modelstorageName)
	}
	return nil
}

func (i *InferenceReconciler) prepareModel(ctx context.Context, modelstorageName string, inference *cpodv1beta1.Inference) error {
	logrus.Info("DEBUG ", "cpodjob", inference.Spec, "Annotations", inference.Annotations)
	modelSize := int64(0)
	modelReadableName := ""
	modelTemplate := ""
	if inference.Annotations != nil {
		if sizeStr, ok := inference.Annotations[cpodv1beta1.CPodPreTrainModelSizeAnno]; ok {
			size, err := strconv.ParseInt(sizeStr, 10, 64)
			if err != nil {
				return fmt.Errorf("failed to parse model size %v: %v", sizeStr, err)
			}
			modelSize = size
		}
		if name, ok := inference.Annotations[v1beta1.CPodPreTrainModelReadableNameAnno]; ok {
			modelReadableName = name
		}

		if template, ok := inference.Annotations[v1beta1.CPodPreTrainModelTemplateAnno]; ok {
			modelTemplate = template
		}
	}
	if inference.Spec.ModelIsPublic {
		modelName := modelstorageName + v1beta1.CPodPublicStorageSuffix
		ms := &cpodv1.ModelStorage{}
		if err := i.Client.Get(ctx, client.ObjectKey{Namespace: inference.Namespace, Name: modelName}, ms); err != nil {
			if apierrors.IsNotFound(err) {
				publicMs := &cpodv1.ModelStorage{}
				if err := i.Client.Get(ctx, client.ObjectKey{Namespace: v1beta1.CPodPublicNamespace, Name: modelstorageName}, publicMs); err != nil {
					if apierrors.IsNotFound(err) {
						if createdMs, err := createModelstorage(ctx, i.Client, modelstorageName, modelReadableName, modelSize, modelTemplate, v1beta1.CPodPublicNamespace, i.Options.StorageClassName); err != nil {
							return fmt.Errorf("failed to create model storage for public model %s: %v", modelstorageName, err)
						} else {
							publicMs = createdMs
						}
					} else {
						return fmt.Errorf("failed to get public model %s: %v", modelstorageName, err)
					}
				}
				if publicMs != nil && publicMs.Status.Phase != "done" {
					jobName := "model-" + modelstorageName
					job := &batchv1.Job{}
					if err := i.Client.Get(ctx, client.ObjectKey{Namespace: v1beta1.CPodPublicNamespace, Name: jobName}, job); err != nil {
						if apierrors.IsNotFound(err) {
							if err := CreateDownloadJob(ctx, i.Client, i.Options.OssOption, "model", modelstorageName, modelReadableName, modelSize, inference.Namespace, v1beta1.CPodPublicNamespace); err != nil {
								return fmt.Errorf("failed to create download job for public model %s: %v", modelstorageName, err)
							}
						} else {
							return fmt.Errorf("failed to get public model %s: %v", modelstorageName, err)
						}
					}
					if job.Status.Succeeded != 1 {
						return fmt.Errorf("public model downloader job %s is running: %v", jobName, job.Status.Succeeded)
					}
					return fmt.Errorf("public model %s is not done", modelstorageName)
				}
				if err := CopyPublicModelStorage(ctx, i.Client, modelstorageName, inference.Namespace); err != nil {
					return fmt.Errorf("failed to copy public model %s: %v", modelstorageName, err)
				}
				return nil
			} else {
				return fmt.Errorf("failed to get public model %v's copy  %s: %v", modelstorageName, modelName, err)
			}
		}
		if ms.Status.Phase != "done" {
			return fmt.Errorf("public model copy  %s is not done", modelstorageName)
		}
		return nil
	}
	ms := &cpodv1.ModelStorage{}
	if err := i.Client.Get(ctx, client.ObjectKey{Namespace: inference.Namespace, Name: modelstorageName}, ms); err != nil {
		if apierrors.IsNotFound(err) {
			if createdMs, err := createModelstorage(ctx, i.Client, modelstorageName, modelReadableName, modelSize, modelTemplate, inference.Namespace, i.Options.StorageClassName); err != nil {
				return fmt.Errorf("failed to create model storage for private model %s: %v", modelstorageName, err)
			} else {
				ms = createdMs
			}
		} else {
			return fmt.Errorf("failed to get private model %s: %v", modelstorageName, err)
		}
	}
	if ms != nil && ms.Status.Phase != "done" {
		jobName := "model-" + modelstorageName
		job := &batchv1.Job{}
		if err := i.Client.Get(ctx, client.ObjectKey{Namespace: inference.Namespace, Name: jobName}, job); err != nil {
			if apierrors.IsNotFound(err) {
				if err := CreateDownloadJob(ctx, i.Client, i.Options.OssOption, "model", modelstorageName, modelReadableName, modelSize, inference.Namespace, inference.Namespace); err != nil {
					return fmt.Errorf("failed to create download job for private model %s: %v", modelstorageName, err)
				}
			} else {
				return fmt.Errorf("failed to get private model %s: %v", modelstorageName, err)
			}
		}
		if job.Status.Succeeded != 1 {
			return fmt.Errorf("model downloader job %s is running: %v", jobName, job.Status.Succeeded)
		}
		return fmt.Errorf("private model %s is not done", modelstorageName)
	}
	return nil
}

func (i *InferenceReconciler) prepareRay(ctx context.Context, inference *cpodv1beta1.Inference) (*rayv1.RayService, error) {
	rayService := &rayv1.RayService{}
	if err := i.Client.Get(ctx, client.ObjectKey{Namespace: inference.Namespace, Name: inference.Name}, rayService); err != nil {
		if apierrors.IsNotFound(err) {
			if err := i.createRayService(ctx, inference); err != nil {
				logrus.Errorf("failed to create ray service %s: %v", inference.Name, err)
				return nil, fmt.Errorf("failed to create ray service %s: %v", inference.Name, err)
			}
			return nil, err
		}
		return nil, fmt.Errorf("failed to get ray service %s: %v", inference.Name, err)
	}

	if err := i.prepareIngressOfRayService(ctx, inference, rayService); err != nil {
		logrus.Errorf("failed to prepare ingress for ray service %s: %v", inference.Name, err)
		return nil, fmt.Errorf("failed to prepare ingress for ray service %s: %v", inference.Name, err)
	}

	return rayService, nil
}

func (i *InferenceReconciler) createRayService(ctx context.Context, inference *cpodv1beta1.Inference) error {
	modelstorageName := inference.Spec.ModelID
	if inference.Spec.ModelIsPublic {
		modelstorageName = modelstorageName + v1beta1.CPodPublicStorageSuffix
	}
	ms := &cpodv1.ModelStorage{}
	if err := i.Client.Get(ctx, client.ObjectKey{Namespace: inference.Namespace, Name: modelstorageName}, ms); err != nil {
		return fmt.Errorf("failed to get model storage %s: %v", modelstorageName, err)
	}

	if ms.Status.Phase != "done" {
		return fmt.Errorf("model storage %s is not ready", modelstorageName)
	}

	minReplicas := int32(1)
	maxReplicas := int32(1)
	if inference.Spec.AutoscalerOptions != nil {
		minReplicas = inference.Spec.AutoscalerOptions.MinReplicas
		if inference.Spec.AutoscalerOptions.MinReplicas == 0 {
			minReplicas = 1
		}
		maxReplicas = inference.Spec.AutoscalerOptions.MaxReplicas
		if inference.Spec.AutoscalerOptions.MaxReplicas == 0 {
			maxReplicas = 1
		}
	}

	rayService := &rayv1.RayService{
		ObjectMeta: metav1.ObjectMeta{
			Name:      inference.Name,
			Namespace: inference.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				i.generateOwnerRefInference(ctx, inference),
			},
		},
		Spec: rayv1.RayServiceSpec{
			ServeConfigV2: fmt.Sprintf(`applications:
    - name: llm
      route_prefix: /
      import_path: vllm_app:model 
      deployments:
      - name: VLLMDeployment
        max_ongoing_requests: 5
        autoscaling_config:
          min_replicas: %v
          initial_replicas: null
          max_replicas: %v
          target_ongoing_requests: 3.0
          metrics_interval_s: 10.0
          look_back_period_s: 30.0
          smoothing_factor: 1.0
          upscale_smoothing_factor: null
          downscale_smoothing_factor: null
          upscaling_factor: null
          downscaling_factor: null
          downscale_delay_s: 600.0
          upscale_delay_s: 30.0
      runtime_env:
        working_dir: "https://sxwl-dg.oss-cn-beijing.aliyuncs.com/ray/ray_vllm/va.zip"`, minReplicas, maxReplicas),
			RayClusterSpec: rayv1.RayClusterSpec{
				EnableInTreeAutoscaling: ptr.To(true),
				AutoscalerOptions: &rayv1.AutoscalerOptions{
					Resources: &corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							"nvidia.com/gpu": resource.MustParse(strconv.FormatInt(int64(inference.Spec.GPUCount), 10)),
						},
						Limits: corev1.ResourceList{
							"nvidia.com/gpu": resource.MustParse(strconv.FormatInt(int64(inference.Spec.GPUCount), 10)),
						},
					},
				},
				HeadGroupSpec: rayv1.HeadGroupSpec{
					RayStartParams: map[string]string{
						"dashboard-host": "0.0.0.0",
					},
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  "ray-head",
									Image: i.Options.RayImage,
									Resources: corev1.ResourceRequirements{
										Requests: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse("2"),
											corev1.ResourceMemory: resource.MustParse("8Gi"),
										},
										Limits: corev1.ResourceList{
											corev1.ResourceCPU:    resource.MustParse("2"),
											corev1.ResourceMemory: resource.MustParse("8Gi"),
										},
									},
									Ports: []corev1.ContainerPort{
										{
											ContainerPort: 6379,
											Name:          "gcs-server",
										},
										{
											ContainerPort: 8265,
											Name:          "dashboard",
										},
										{
											ContainerPort: 10001,
											Name:          "client",
										},
										{
											ContainerPort: 8000,
											Name:          "serve",
										},
									},
								},
							},
						},
					},
				},
				WorkerGroupSpecs: []rayv1.WorkerGroupSpec{
					{
						Replicas:    &minReplicas,
						MinReplicas: &minReplicas,
						MaxReplicas: &maxReplicas,
						GroupName:   "gpu-group",
						RayStartParams: map[string]string{
							"num-cpus": fmt.Sprintf("%d", inference.Spec.GPUCount),
						},
						Template: corev1.PodTemplateSpec{
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{
										Name:  "llm",
										Image: i.Options.RayImage,
										Env:   []corev1.EnvVar{},

										Resources: corev1.ResourceRequirements{
											Requests: corev1.ResourceList{
												corev1.ResourceCPU:    resource.MustParse("4"),
												corev1.ResourceMemory: resource.MustParse("20Gi"),
												"nvidia.com/gpu":      resource.MustParse(strconv.FormatInt(int64(inference.Spec.GPUCount), 10)),
											},
											Limits: corev1.ResourceList{
												corev1.ResourceCPU:    resource.MustParse("4"),
												corev1.ResourceMemory: resource.MustParse("20Gi"),
												"nvidia.com/gpu":      resource.MustParse(strconv.FormatInt(int64(inference.Spec.GPUCount), 10)),
											},
										},
										VolumeMounts: []corev1.VolumeMount{
											{
												Name:      "cache-volume",
												MountPath: "/dev/shm",
											},
											{
												Name:      "model",
												MountPath: "/mnt/models",
											},
										},
									},
								},
								Volumes: []corev1.Volume{
									{
										Name: "cache-volume",
										VolumeSource: corev1.VolumeSource{
											EmptyDir: &corev1.EmptyDirVolumeSource{
												Medium:    corev1.StorageMediumMemory,
												SizeLimit: resource.NewQuantity(20*1024*1024*1024, resource.BinarySI),
											},
										},
									},
									{
										Name: "model",
										VolumeSource: corev1.VolumeSource{
											PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
												ClaimName: ms.Spec.PVC,
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	if inference.Spec.Params != nil {
		rayService.Spec.RayClusterSpec.WorkerGroupSpecs[0].Template.Spec.Containers[0].Env = append(rayService.Spec.RayClusterSpec.WorkerGroupSpecs[0].Template.Spec.Containers[0].Env, corev1.EnvVar{
			Name:  "EXTRA_PARAMS",
			Value: *inference.Spec.Params,
		})
	}

	if inference.Spec.GPUCount > 1 {
		rayService.Spec.RayClusterSpec.WorkerGroupSpecs[0].Template.Spec.Containers[0].Env = append(rayService.Spec.RayClusterSpec.WorkerGroupSpecs[0].Template.Spec.Containers[0].Env, corev1.EnvVar{
			Name:  "TENSOR_PARALLELISM",
			Value: strconv.Itoa(inference.Spec.GPUCount),
		})
	}

	if err := i.Client.Create(ctx, rayService); err != nil {
		return fmt.Errorf("failed to create ray service %s: %v", inference.Name, err)
	}
	return nil
}

func (i *InferenceReconciler) prepareKserve(ctx context.Context, inference *cpodv1beta1.Inference) (*kservev1beta1.InferenceService, error) {
	inferenceService := &kservev1beta1.InferenceService{}
	if err := i.Client.Get(ctx, client.ObjectKey{Namespace: inference.Namespace, Name: i.getInferenceServiceName(inference)}, inferenceService); err != nil {
		if apierrors.IsNotFound(err) {
			if err := i.createInferenceService(ctx, inference); err != nil {
				return nil, fmt.Errorf("failed to create inference service %s: %v", inference.Name, err)
			}
			return nil, err
		}
		return nil, fmt.Errorf("failed to get inference service %s: %v", inference.Name, err)
	}

	if err := i.prepareIngressOfInferenceService(ctx, inference, inferenceService); err != nil {
		return nil, fmt.Errorf("failed to prepare ingress for inference service %s: %v", inference.Name, err)
	}
	return inferenceService, nil
}

func (i *InferenceReconciler) createInferenceService(ctx context.Context, inference *cpodv1beta1.Inference) error {
	if len(inference.Spec.Predictor.GetImplementations()) == 0 {
		return fmt.Errorf("the implementation of predictor is null")
	}
	predictor := inference.Spec.Predictor.GetImplementation()
	if sourceURI := predictor.GetStorageUri(); sourceURI != nil {
		if strings.HasPrefix(*sourceURI, cpodv1beta1.ModelStoragePrefix) {
			modelstorageName := inference.Spec.ModelID
			if inference.Spec.ModelIsPublic {
				modelstorageName = modelstorageName + cpodv1beta1.CPodPublicStorageSuffix
			}
			modelstorage := cpodv1.ModelStorage{}
			if err := i.Client.Get(ctx, client.ObjectKey{
				Namespace: inference.Namespace,
				Name:      modelstorageName,
			}, &modelstorage); err != nil {
				if apierrors.IsNotFound(err) {
					// TODO: 更新condition
					i.Recorder.Eventf(inference, corev1.EventTypeWarning, "GetModelstorageFailed", "modelstorage not found")
					return err
				}
				return err
			}

			if modelstorage.Status.Phase != "done" {
				return fmt.Errorf("modelstorage %s is not ready", modelstorage.Name)
			}

			pre := cpodv1beta1.Predictor(inference.Spec.Predictor)
			pre.SetStorageURI("pvc://" + modelstorage.Spec.PVC)

		}
	}

	is := kservev1beta1.InferenceService{
		ObjectMeta: metav1.ObjectMeta{
			Name:      inference.Name + "-is",
			Namespace: inference.Namespace,
			Labels:    inference.Labels,
			OwnerReferences: []metav1.OwnerReference{
				i.generateOwnerRefInference(ctx, inference),
			},
		},
		Spec: kservev1beta1.InferenceServiceSpec{
			Predictor: inference.Spec.Predictor,
		},
	}

	if metav1.HasAnnotation(inference.ObjectMeta, cpodv1beta1.CPodAdapterIDAnno) {
		adapterID := inference.Annotations[cpodv1beta1.CPodAdapterIDAnno]
		if metav1.HasAnnotation(inference.ObjectMeta, cpodv1beta1.CPodAdapterIsPublic) {
			adapterID = adapterID + v1beta1.CPodPublicStorageSuffix
		}
		adapterMs := cpodv1.ModelStorage{}
		if err := i.Client.Get(ctx, client.ObjectKey{Namespace: inference.Namespace, Name: adapterID}, &adapterMs); err != nil {
			if apierrors.IsNotFound(err) {
				// TODO: 更新condition
				i.Recorder.Eventf(inference, corev1.EventTypeWarning, "GetAdatperFailed", "adatper modelstorage not found")
				return err
			}
			return err
		}

		if adapterMs.Status.Phase != "done" {
			return fmt.Errorf("adapter %s is not ready", adapterMs.Name)
		}

		is.Spec.Predictor.Volumes = append(is.Spec.Predictor.Volumes, corev1.Volume{
			Name: "adapter",
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: adapterMs.Spec.PVC,
				},
			},
		})

		is.Spec.Predictor.Containers[0].VolumeMounts = append(is.Spec.Predictor.Containers[0].VolumeMounts, corev1.VolumeMount{
			Name:      "adapter",
			MountPath: "/mnt/adapter",
		})

		is.Spec.Predictor.Containers[0].Command = append(is.Spec.Predictor.Containers[0].Command, []string{"--adapter_name_or_path", "/mnt/adapter"}...)
	}

	return i.Client.Create(ctx, &is)
}

func (i *InferenceReconciler) prepareWebUI(ctx context.Context, inference *cpodv1beta1.Inference) error {
	apiURL := fmt.Sprintf("http://%s.%s.svc.cluster.local/v1/chat/completions", PredictorServiceName(i.getInferenceServiceName(inference)), inference.Namespace)
	if inference.Spec.Backend == cpodv1beta1.InferenceBackendRay {
		// 获取对应preidector service name
		svcName := inference.Name + "-serve-svc"
		// 如果 svcName 的长度大于 50，则截取后 50 个字符
		if len(svcName) > 50 {
			svcName = svcName[len(svcName)-50:]
		}
		apiURL = fmt.Sprintf("http://%s.%s.svc.cluster.local:8000/v1/chat/completions", svcName, inference.Namespace)
	}
	webUIDeployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      inference.Name + "-web-ui",
			Namespace: inference.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				i.generateOwnerRefInference(ctx, inference),
			},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: ptr.To(int32(1)),
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app": inference.Name + "-web-ui",
				},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app": inference.Name + "-web-ui",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  inference.Name + "-web-ui",
							Image: i.Options.InferenceWebuiImage,
							Ports: []corev1.ContainerPort{
								{
									ContainerPort: 8000,
								},
							},
							Env: []corev1.EnvVar{
								{
									Name:  "API_URL",
									Value: apiURL,
								},
								{
									Name:  "ENV_BASE_URL",
									Value: fmt.Sprintf("/inference/%s/", inference.Name),
								},
							},
						},
					},
				},
			},
		},
	}
	if err := i.Client.Get(ctx, client.ObjectKey{Namespace: inference.Namespace, Name: inference.Name + "-web-ui"}, webUIDeployment); err != nil {
		if apierrors.IsNotFound(err) {
			return i.Client.Create(ctx, webUIDeployment)
		}
		return fmt.Errorf("failed to get web ui deployment %s: %v", inference.Name+"-web-ui", err)
	}

	webUIService := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      inference.Name + "-web-ui",
			Namespace: inference.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				i.generateOwnerRefInference(ctx, inference),
			},
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{
				"app": inference.Name + "-web-ui",
			},
			Ports: []corev1.ServicePort{
				{
					Port:       8000,
					TargetPort: intstr.FromInt(8000),
				},
			},
		},
	}
	if err := i.Client.Get(ctx, client.ObjectKey{Namespace: inference.Namespace, Name: inference.Name + "-web-ui"}, webUIService); err != nil {
		if apierrors.IsNotFound(err) {
			return i.Client.Create(ctx, webUIService)
		}
		return fmt.Errorf("failed to get web ui service %s: %v", inference.Name+"-web-ui", err)
	}

	pathType := netv1.PathTypeImplementationSpecific
	webUIIngressName := inference.Name + "-web-ui-ingress"
	webUISvcName := inference.Name + "-web-ui"
	webUIPort := int32(8000)

	webUIIngress := &netv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:      webUIIngressName,
			Namespace: inference.Namespace,
			OwnerReferences: []metav1.OwnerReference{
				i.generateOwnerRefInference(ctx, inference),
			},
			Annotations: map[string]string{
				"nginx.ingress.kubernetes.io/rewrite-target": "/$2",
			},
		},
		Spec: netv1.IngressSpec{
			IngressClassName: ptr.To("nginx"),
			Rules: []netv1.IngressRule{
				{
					IngressRuleValue: netv1.IngressRuleValue{
						HTTP: &netv1.HTTPIngressRuleValue{
							Paths: []netv1.HTTPIngressPath{
								{
									PathType: &pathType,
									Path:     "/inference/" + inference.Name + "(/|$)(.*)",
									Backend: netv1.IngressBackend{
										Service: &netv1.IngressServiceBackend{
											Name: webUISvcName,
											Port: netv1.ServiceBackendPort{
												Number: webUIPort,
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
	if err := i.Client.Get(ctx, client.ObjectKey{Namespace: inference.Namespace, Name: webUIIngressName}, webUIIngress); err != nil {
		if apierrors.IsNotFound(err) {
			return i.Client.Create(ctx, webUIIngress)
		}
		return fmt.Errorf("failed to get web ui ingress %s: %v", webUIIngressName, err)
	}

	return nil
}
