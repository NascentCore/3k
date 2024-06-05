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

package controller

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"knative.dev/pkg/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	cpodv1 "github.com/NascentCore/cpodoperator/api/v1"
	"github.com/NascentCore/cpodoperator/api/v1beta1"
	cpodv1beta1 "github.com/NascentCore/cpodoperator/api/v1beta1"
	"github.com/NascentCore/cpodoperator/pkg/util"

	commonv1 "github.com/kubeflow/common/pkg/apis/common/v1"
	"github.com/sirupsen/logrus"

	mpiv2 "github.com/kubeflow/mpi-operator/pkg/apis/kubeflow/v2beta1"
	tov1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	toutil "github.com/kubeflow/training-operator/pkg/util"
)

type OssOption struct {
	DownloaderImage string
	OssAK, OssAS    string
	BucketName      string
}

type CPodJobOption struct {
	StorageClassName           string
	ModelUploadJobImage        string
	ModelUploadJobBackoffLimit int32
	ModelUploadOssBucketName   string
	OssOption                  OssOption
}

// CPodJobReconciler reconciles a CPodJob object
type CPodJobReconciler struct {
	client.Client
	Scheme *runtime.Scheme

	// Recorder is an event recorder for recording Event resources to the
	// Kubernetes API.
	Recorder record.EventRecorder

	Option *CPodJobOption
}

//+kubebuilder:rbac:groups=cpod.cpod,resources=cpodjobs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=cpod.cpod,resources=cpodjobs/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=cpod.cpod,resources=cpodjobs/finalizers,verbs=update
//+kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=cpod.cpod,resources=modelstorages,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=cpod.cpod,resources=datasetstorages,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups="core",resources=persistentvolumeclaims,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups="core",resources=persistentvolumes,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups="core",resources=secrets,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups="core",resources=configmaps,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups="core",resources=events,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups="core",resources=nodes,verbs=get;list;watch
//+kubebuilder:rbac:groups=kubeflow.org,resources=mpijobs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=kubeflow.org,resources=mpijobs/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=kubeflow.org,resources=pytorchjobs,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=kubeflow.org,resources=pytorchjobs/status,verbs=get;update;patch

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the CPodJob object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.16.3/pkg/reconcile
func (c *CPodJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (_ ctrl.Result, reterr error) {
	logger := log.FromContext(ctx)

	cpodjob := &cpodv1beta1.CPodJob{}
	if err := c.Client.Get(ctx, req.NamespacedName, cpodjob); client.IgnoreNotFound(err) != nil {
		logger.Error(err, "unable to fetch CPodJob")
		return ctrl.Result{}, err
	}

	if !c.needReconcile(cpodjob) {
		if err := c.releaseSavedModel(ctx, cpodjob); err != nil {
			return ctrl.Result{}, err
		}

		if err := c.createGeneratedModelstorage(ctx, cpodjob); err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	}

	oldCpodjobStatus := cpodjob.Status.DeepCopy()

	defer func() {
		if !equality.Semantic.DeepEqual(oldCpodjobStatus, &cpodjob.Status) {
			if err := c.Client.Status().Update(ctx, cpodjob); err != nil {
				logger.Error(err, "unable to update CPodJob status")
				reterr = err
			}
		}
	}()

	if !util.IsFinshed(cpodjob.Status) {
		// 判断模型和数据集是否已经准备完毕
		prepareCond := util.GetCondition(cpodjob.Status, cpodv1beta1.JobDataPreparing)
		if prepareCond == nil || prepareCond.Status != corev1.ConditionTrue {
			if err := c.PrepareData(ctx, cpodjob); err != nil {
				logger.Error(err, "unable to prepare data", "cpdojob", cpodjob)
				util.UpdateJobConditions(&cpodjob.Status, cpodv1beta1.JobDataPreparing, corev1.ConditionFalse, "PrepareDate", err.Error())
				return ctrl.Result{}, err
			}
		}
		util.UpdateJobConditions(&cpodjob.Status, cpodv1beta1.JobDataPreparing, corev1.ConditionTrue, "DataReady", "Data is ready")

		baseJob, err := c.GetBaseJob(ctx, cpodjob)
		if err != nil {
			if apierrors.IsNotFound(err) {
				err = c.CreateBaseJob(ctx, cpodjob)
				if err != nil {
					logger.Error(err, "unable to create baseJob")
					return ctrl.Result{}, err
				}
				// TODO: @sxwl-donggang update the condition of cpodjob
				return ctrl.Result{Requeue: true}, nil
			}
			return ctrl.Result{}, err
		}

		// 判断baseJob有没有到达稳定态
		baseJobStatus := c.GetBaseJobStatus(ctx, cpodjob, baseJob)
		if baseJobStatus == nil {
			logger.Info("baseJobStatus is nil")
			return ctrl.Result{Requeue: true}, nil
		}

		if err := c.UpdateStatus(ctx, cpodjob, baseJobStatus); err != nil {
			logger.Error(err, "unable to update CPodJob status")
			return ctrl.Result{}, err
		}
	}

	if err := c.releaseSavedModel(ctx, cpodjob); err != nil {
		return ctrl.Result{}, err
	}

	if err := c.createGeneratedModelstorage(ctx, cpodjob); err != nil {
		return ctrl.Result{}, err
	}

	if util.IsSucceeded(cpodjob.Status) && cpodjob.Spec.UploadModel && cpodjob.Spec.ModelSavePath != "" {
		var userID, jobName string
		var ok bool
		if userID, ok = cpodjob.Labels[v1beta1.CPodUserIDLabel]; !ok {
			return ctrl.Result{}, nil
		}
		jobName = cpodjob.Name
		if strings.HasSuffix(jobName, "-cpodjob") {
			jobName, _ = strings.CutSuffix(jobName, "-cpodjob")
		}

		err := c.uploadSavedModel(ctx, cpodjob, userID, jobName)
		if err != nil {
			return ctrl.Result{}, err
		}
	}

	// TODO: @sxwl-donggang 如果cpodjob执行成功，modelsavepath不为空，删除cpodjob不删除对应modelsave pvc

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (c *CPodJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	// watch to events for cpodjob and its owned subresources basejob and pvc
	return ctrl.NewControllerManagedBy(mgr).
		For(&cpodv1beta1.CPodJob{}, builder.WithPredicates(
			predicate.Funcs{CreateFunc: c.onOwnerCreateFunc()},
		)).
		Owns(&mpiv2.MPIJob{}).
		Complete(c)
}

func (c *CPodJobReconciler) needReconcile(cpodjob *cpodv1beta1.CPodJob) bool {
	if util.IsFailed(cpodjob.Status) {
		return false
	}
	if util.IsSucceeded(cpodjob.Status) {
		if cpodjob.Spec.UploadModel && cpodjob.Spec.ModelSavePath != "" {
			if cond := util.GetCondition(cpodjob.Status, cpodv1beta1.JobModelUploaded); cond != nil {
				return false
			}
			return true
		}
		return false
	}

	return true
}

// CreateBaseJob creates the base job object based on the job type specified in the CPodJob.
func (c *CPodJobReconciler) CreateBaseJob(ctx context.Context, cpodjob *cpodv1beta1.CPodJob) error {
	// 需要判断是否使用分布式训练，尽可能在节点运行，考虑以下因素
	// 1. 用户制定，需要清楚训练任务是否支持分布式训练
	// 2. 节点GPU使用数量
	// 3. 显存
	// 4. GPU型号：
	//    * 训练任务不允许使用不同信号的GPU;
	// 5. 网络：
	//     * 分布式训练任务

	logger := log.FromContext(ctx)
	volumes := []corev1.Volume{}
	volumeMounts := []corev1.VolumeMount{}

	addVolume := func(name, claimName, mountPath string) {
		volumes = append(volumes, corev1.Volume{
			Name: name,
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: claimName,
					ReadOnly:  false,
				},
			},
		})
		volumeMounts = append(volumeMounts, corev1.VolumeMount{
			Name:      name,
			MountPath: mountPath,
		})
	}

	// 检查 logs PVC 是否存在
	exists, err := c.checkPVCExists(ctx, "logs", cpodjob.Namespace)
	if err != nil {
		logger.Info("Warning: checking if 'logs' PVC exists: %v", err)
	}
	if !exists {
		logger.Info("Warning:'logs' PVC does not exist in namespace %s", "namespace", cpodjob.Namespace)
	} else {
		addVolume("logs", "logs", "/logs")
	}

	if cpodjob.Spec.CKPTPath != "" && cpodjob.Spec.CKPTVolumeSize != 0 {
		ckptPVC, err := c.GetCKPTPVC(ctx, cpodjob)
		if err != nil {
			c.Recorder.Eventf(cpodjob, corev1.EventTypeWarning, "GetCKPTPVCFailed", "Get ckpt pvc failed")
			return err
		}
		addVolume("ckpt", ckptPVC.Name, cpodjob.Spec.CKPTPath)
	}

	if cpodjob.Spec.ModelSavePath != "" && cpodjob.Spec.ModelSaveVolumeSize != 0 {
		modelSavePVC, err := c.GetModelSavePVC(ctx, cpodjob)
		if err != nil {
			c.Recorder.Eventf(cpodjob, corev1.EventTypeWarning, "GetModelSavePVCFailed", "Get model save pvc failed")
			return err
		}
		addVolume("modelsave", modelSavePVC.Name, cpodjob.Spec.ModelSavePath)
	}

	if cpodjob.Spec.DatasetPath != "" && cpodjob.Spec.DatasetName != "" {
		datasetName := cpodjob.Spec.DatasetName
		if cpodjob.Spec.DatasetIsPublic {
			datasetName = datasetName + v1beta1.CPodPublicStorageSuffix
		}
		dataset := &cpodv1.DataSetStorage{}
		if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: datasetName}, dataset); err != nil {
			if apierrors.IsNotFound(err) {
				c.Recorder.Eventf(cpodjob, corev1.EventTypeWarning, "DatasetFailed", "Dataset not found")
				util.UpdateJobConditions(&cpodjob.Status, cpodv1beta1.JobFailed, corev1.ConditionTrue, "GetDatasetFailed", "Dataset not found")
				return c.UpdateStatus(ctx, cpodjob, nil)
			}
			return err
		}
		if dataset.Status.Phase != "done" {
			c.Recorder.Eventf(cpodjob, corev1.EventTypeWarning, "GetDatasetFailed", "Dateset not found")
			util.UpdateJobConditions(&cpodjob.Status, cpodv1beta1.JobFailed, corev1.ConditionTrue, "GetPretrainModelFailed", "Pretrain model downloader failed")
			return c.UpdateStatus(ctx, cpodjob, nil)
		}

		datasetPVC := &corev1.PersistentVolumeClaim{}
		if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: dataset.Spec.PVC}, datasetPVC); err != nil {
			if apierrors.IsNotFound(err) {
				c.Recorder.Eventf(cpodjob, corev1.EventTypeWarning, "GetDatasetPVCFailed", "Dataset PVC not found")
				util.UpdateJobConditions(&cpodjob.Status, cpodv1beta1.JobFailed, corev1.ConditionTrue, "GetDatasetPVCFailed", "Dataset PVC not found")
				return c.UpdateStatus(ctx, cpodjob, nil)
			}
			return err
		}
		addVolume("dataset", datasetPVC.Name, cpodjob.Spec.DatasetPath)
	}

	if cpodjob.Spec.PretrainModelName != "" && cpodjob.Spec.PretrainModelPath != "" {
		modelName := cpodjob.Spec.PretrainModelName
		if cpodjob.Spec.PretrainModelIsPublic {
			modelName = modelName + v1beta1.CPodPublicStorageSuffix
		}
		pretrainModel := &cpodv1.ModelStorage{}
		if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: modelName}, pretrainModel); err != nil {
			if apierrors.IsNotFound(err) {
				c.Recorder.Eventf(cpodjob, corev1.EventTypeWarning, "GetPretrainModelFailed", "Pretrain model not found")
				util.UpdateJobConditions(&cpodjob.Status, cpodv1beta1.JobFailed, corev1.ConditionTrue, "GetPretrainModelFailed", "Pretrain model not found")
				return c.UpdateStatus(ctx, cpodjob, nil)
			}
			return err
		}
		if pretrainModel.Status.Phase != "done" {
			c.Recorder.Eventf(cpodjob, corev1.EventTypeWarning, "GetPretrainModelFailed", "Pretrain model not found")
			util.UpdateJobConditions(&cpodjob.Status, cpodv1beta1.JobFailed, corev1.ConditionTrue, "GetPretrainModelFailed", "Pretrain model downloader failed")
			return c.UpdateStatus(ctx, cpodjob, nil)
		}

		pretrainModelPVC := &corev1.PersistentVolumeClaim{}
		if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: pretrainModel.Spec.PVC}, pretrainModelPVC); err != nil {
			if apierrors.IsNotFound(err) {
				c.Recorder.Eventf(cpodjob, corev1.EventTypeWarning, "GetPretrainModelPVCFailed", "Pretrain model PVC not found")
				util.UpdateJobConditions(&cpodjob.Status, cpodv1beta1.JobFailed, corev1.ConditionTrue, "GetPretrainModelPVCFailed", "Pretrain model PVC not found")
				return c.UpdateStatus(ctx, cpodjob, nil)
			}
		}
		addVolume("pretrain-model", pretrainModelPVC.Name, cpodjob.Spec.PretrainModelPath)
	}

	// Set replicas based on worker replicas
	workerReplicas := int32(1)
	if cpodjob.Spec.WorkerReplicas != 0 {
		workerReplicas = int32(cpodjob.Spec.WorkerReplicas)
	}

	resources := corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceName("nvidia.com/gpu"): resource.MustParse(strconv.Itoa(int(cpodjob.Spec.GPURequiredPerReplica))),
		},
		Limits: corev1.ResourceList{
			corev1.ResourceName("nvidia.com/gpu"): resource.MustParse(strconv.Itoa(int(cpodjob.Spec.GPURequiredPerReplica))),
		},
	}
	if c.needAllocateRDMADevice(ctx, cpodjob) {
		resources.Requests["rdma/rdma_shared_device_a"] = *resource.NewQuantity(1, resource.DecimalSI)
		resources.Limits["rdma/rdma_shared_device_a"] = *resource.NewQuantity(1, resource.DecimalSI)
	}

	// Create target job based on job type
	var targetJob client.Object

	switch cpodjob.Spec.JobType {
	case cpodv1beta1.JobTypeMPI:
		targetJob = &mpiv2.MPIJob{
			ObjectMeta: metav1.ObjectMeta{
				Name:      cpodjob.Name,
				Namespace: cpodjob.Namespace,
				OwnerReferences: []metav1.OwnerReference{
					c.generateOwnerRefCPodJob(ctx, cpodjob),
				},
			},
			Spec: mpiv2.MPIJobSpec{
				MPIReplicaSpecs: map[mpiv2.MPIReplicaType]*commonv1.ReplicaSpec{
					mpiv2.MPIReplicaTypeLauncher: {
						RestartPolicy: commonv1.RestartPolicyOnFailure,
						Template: corev1.PodTemplateSpec{
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{
										Env:          cpodjob.Spec.Envs,
										Name:         "launcher",
										Command:      cpodjob.Spec.Command,
										Args:         cpodjob.Spec.Args,
										Image:        cpodjob.Spec.Image,
										VolumeMounts: volumeMounts,
									},
								},
								HostIPC: true,
								Volumes: volumes,
							},
						},
					},
					mpiv2.MPIReplicaTypeWorker: {
						Replicas:      &workerReplicas,
						RestartPolicy: commonv1.RestartPolicyOnFailure,
						Template: corev1.PodTemplateSpec{
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{
										Env:          cpodjob.Spec.Envs,
										Image:        cpodjob.Spec.Image,
										Name:         "worker",
										VolumeMounts: volumeMounts,
										Resources:    resources,
									},
								},
								Volumes: volumes,
								NodeSelector: map[string]string{
									"nvidia.com/gpu.product": cpodjob.Spec.GPUType,
								},
							},
						},
					},
				},

				// RunPolicy: mpiv2.RunPolicy(runpolicy),
			},
		}
	case cpodv1beta1.JobTypePytorch:
		backendC10D := tov1.BackendC10D
		targetJob = &tov1.PyTorchJob{
			ObjectMeta: metav1.ObjectMeta{
				Name:      cpodjob.Name,
				Namespace: cpodjob.Namespace,
				OwnerReferences: []metav1.OwnerReference{
					c.generateOwnerRefCPodJob(ctx, cpodjob),
				},
			},
			Spec: tov1.PyTorchJobSpec{
				RunPolicy: tov1.RunPolicy{
					CleanPodPolicy: tov1.CleanPodPolicyPointer(tov1.CleanPodPolicyRunning),
					BackoffLimit:   cpodjob.Spec.BackoffLimit,
				},
				PyTorchReplicaSpecs: map[tov1.ReplicaType]*tov1.ReplicaSpec{
					tov1.PyTorchJobReplicaTypeWorker: {
						Replicas:      &workerReplicas,
						RestartPolicy: tov1.RestartPolicyOnFailure,
						Template: corev1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Name:      "pytorch-worker",
								Namespace: cpodjob.Namespace,
							},
							Spec: corev1.PodSpec{
								Containers: []corev1.Container{
									{
										Name:         "pytorch",
										Env:          cpodjob.Spec.Envs,
										Image:        cpodjob.Spec.Image,
										Command:      cpodjob.Spec.Command,
										Args:         cpodjob.Spec.Args,
										VolumeMounts: volumeMounts,
										Resources:    resources,
									},
								},
								Volumes: volumes,
								NodeSelector: map[string]string{
									"nvidia.com/gpu.product": cpodjob.Spec.GPUType,
								},
							},
						},
					},
				},
			},
		}
		// it is a distributed training job
		if cpodjob.Spec.WorkerReplicas > 1 {
			targetJobSpec := targetJob.(*tov1.PyTorchJob)
			targetJobSpec.Spec.ElasticPolicy = &tov1.ElasticPolicy{
				RDZVBackend: &backendC10D,
			}
			nprocPerNode := strconv.Itoa(int(cpodjob.Spec.WorkerReplicas))
			targetJobSpec.Spec.NprocPerNode = &nprocPerNode
			workerSpec := targetJobSpec.Spec.PyTorchReplicaSpecs[tov1.PaddleJobReplicaTypeWorker].Template.Spec
			workerSpec.Containers[0].Env = append(workerSpec.Containers[0].Env, []corev1.EnvVar{
				{
					Name:  "NCCL_NET",
					Value: "IB",
				},
				{
					Name:  "NCCL_IB_DISABLE",
					Value: "0",
				},
			}...)
			workerSpec.Containers[0].SecurityContext = &corev1.SecurityContext{
				Capabilities: &corev1.Capabilities{
					Add: []corev1.Capability{
						"IPC_LOCK",
					},
				},
			}
		}

		if cpodjob.Spec.GPURequiredPerReplica > 1 {
			targetJobSpec := targetJob.(*tov1.PyTorchJob)
			workerSpec := targetJobSpec.Spec.PyTorchReplicaSpecs[tov1.PaddleJobReplicaTypeWorker].Template.Spec

			volumes := workerSpec.Volumes
			volumes = append(volumes, corev1.Volume{
				Name: "shm",
				VolumeSource: corev1.VolumeSource{
					EmptyDir: &corev1.EmptyDirVolumeSource{
						Medium:    corev1.StorageMediumMemory,
						SizeLimit: resource.NewQuantity(5120*1024*1024, resource.BinarySI),
					},
				},
			})
			targetJobSpec.Spec.PyTorchReplicaSpecs[tov1.PaddleJobReplicaTypeWorker].Template.Spec.Volumes = volumes

			workerSpec.Containers[0].VolumeMounts = append(workerSpec.Containers[0].VolumeMounts, corev1.VolumeMount{
				Name:      "shm",
				MountPath: "/dev/shm",
			})
		}
	}

	return client.IgnoreAlreadyExists(c.Client.Create(ctx, targetJob))
}

func (c *CPodJobReconciler) checkPVCExists(ctx context.Context, pvcName string, namespace string) (bool, error) {
	pvc := &corev1.PersistentVolumeClaim{}
	err := c.Client.Get(ctx, types.NamespacedName{Name: pvcName, Namespace: namespace}, pvc)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		return false, err
	}

	return true, nil
}

func (c *CPodJobReconciler) UpdateStatus(ctx context.Context, cpodjob *cpodv1beta1.CPodJob, baseJobStatus *tov1.JobStatus) error {
	if cpodjob.Status.StartTime == nil {
		now := metav1.Now()
		cpodjob.Status.StartTime = &now
	}

	if baseJobStatus != nil {
		if toutil.IsFailed(*baseJobStatus) {
			baseJobFailedCond := util.GetBaseJobCondition(*baseJobStatus, tov1.JobFailed)
			util.UpdateJobConditions(&cpodjob.Status, cpodv1beta1.JobFailed, corev1.ConditionTrue, "BaseJobFailed", baseJobFailedCond.Message)
		} else if toutil.IsSucceeded(*baseJobStatus) {
			baseJobSucceedCond := util.GetBaseJobCondition(*baseJobStatus, tov1.JobSucceeded)
			util.UpdateJobConditions(&cpodjob.Status, cpodv1beta1.JobSucceeded, corev1.ConditionTrue, baseJobSucceedCond.Reason, baseJobSucceedCond.Message)
			cpodjob.Status.CompletionTime = baseJobStatus.CompletionTime
		} else {
			util.UpdateJobConditions(&cpodjob.Status, cpodv1beta1.JobRunning, corev1.ConditionTrue, "BaseJobRunning", "BaseJob is running")
			return fmt.Errorf("baseJob is running")
		}
	}

	return nil
}

// GetBaseJob retrieves the base job object based on the job type specified in the CPodJob.
// It returns the target job object and an error, if any.
func (c *CPodJobReconciler) GetBaseJob(ctx context.Context, cpodjob *cpodv1beta1.CPodJob) (client.Object, error) {
	var targetJob client.Object
	switch cpodjob.Spec.JobType {
	case cpodv1beta1.JobTypeMPI:
		targetJob = &mpiv2.MPIJob{}
	case cpodv1beta1.JobTypePytorch:
		targetJob = &tov1.PyTorchJob{}
	}
	return targetJob, c.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: cpodjob.Name}, targetJob)
}

// 由于MPIJob使用的是mpi-controller中的定义，与training-operator的定义不一致，需要进行转换
func (c *CPodJobReconciler) GetBaseJobStatus(ctx context.Context, cpodjob *cpodv1beta1.CPodJob, baseJob client.Object) *tov1.JobStatus {
	switch cpodjob.Spec.JobType {
	case cpodv1beta1.JobTypePytorch:
		pytJob := baseJob.(*tov1.PyTorchJob)
		return &pytJob.Status
	case cpodv1beta1.JobTypeMPI:
		mpiJob := baseJob.(*mpiv2.MPIJob)
		return &tov1.JobStatus{
			Conditions: c.ConvertMPIJobConditionToCommonJobCondition(ctx, mpiJob.Status.Conditions),
			// TODO: @sxwl-donggang ReplicaStatuses is currently not used, consider if it needs to be used in the future
			StartTime:         mpiJob.Status.StartTime,
			LastReconcileTime: mpiJob.Status.LastReconcileTime,
			CompletionTime:    mpiJob.Status.CompletionTime,
		}
	default:
		return nil
	}
}

// 将mpiv1.JobCondition转换为commonv1.JobCondition
func (c *CPodJobReconciler) ConvertMPIJobConditionToCommonJobCondition(ctx context.Context, mpiJobConditions []mpiv2.JobCondition) []tov1.JobCondition {
	res := make([]tov1.JobCondition, len(mpiJobConditions))
	for i, mpiJobCondition := range mpiJobConditions {
		res[i] = tov1.JobCondition{
			Type:               tov1.JobConditionType(mpiJobCondition.Type),
			Status:             mpiJobCondition.Status,
			LastUpdateTime:     mpiJobCondition.LastUpdateTime,
			LastTransitionTime: mpiJobCondition.LastTransitionTime,
			Reason:             mpiJobCondition.Reason,
			Message:            mpiJobCondition.Message,
		}
	}
	return res
}

// GetCKPTPVC retrieves the PersistentVolumeClaim (PVC) associated with the given CPodJob.
// If the PVC does not exist, it will be created and associated with the CPodJob.
// If the PVC is not bound, the function will return an error.
// Parameters:
//   - ctx: The context.Context object for the request.
//   - cpodjob: The CPodJob object for which to retrieve the PVC.
//
// Returns:
//   - *corev1.PersistentVolumeClaim: The retrieved or created PVC.
//   - error: An error if the PVC retrieval or creation fails, or if the PVC is not bound.
func (c *CPodJobReconciler) GetCKPTPVC(ctx context.Context, cpodjob *cpodv1beta1.CPodJob) (*corev1.PersistentVolumeClaim, error) {
	logger := log.FromContext(ctx)
	ckptPVCName := cpodjob.Name + "-ckpt"
	pvc := &corev1.PersistentVolumeClaim{}
	if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: ckptPVCName}, pvc); err != nil {
		if apierrors.IsNotFound(err) {
			logger.Info("ckpt pvc not found, create it")
			volumeMode := corev1.PersistentVolumeFilesystem
			createErr := c.Client.Create(ctx, &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      ckptPVCName,
					Namespace: cpodjob.Namespace,
					OwnerReferences: []metav1.OwnerReference{
						c.generateOwnerRefCPodJob(ctx, cpodjob),
					},
				},
				Spec: corev1.PersistentVolumeClaimSpec{
					AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteMany},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceStorage: *resource.NewQuantity((int64(cpodjob.Spec.CKPTVolumeSize) * 1024 * 1024), resource.BinarySI),
						},
					},
					StorageClassName: &c.Option.StorageClassName,
					VolumeMode:       &volumeMode,
				},
			})
			if createErr != nil {
				logger.Error(createErr, "create ckpt pvc failed")
				return nil, createErr
			}
			// TODO: @sxwl-donggang Should not return an error
			return nil, err
		}
		return nil, err
	}

	// Check if the PVC is bound
	if pvc.Status.Phase != corev1.ClaimBound {
		logger.Info("ckpt pvc not bound, wait for binding")
		return nil, fmt.Errorf("ckpt pvc not bound, wait for binding")
	}
	return pvc, nil
}

func (c *CPodJobReconciler) GetModelSavePVC(ctx context.Context, cpodjob *cpodv1beta1.CPodJob) (*corev1.PersistentVolumeClaim, error) {
	logger := log.FromContext(ctx)
	modeSavePVCName := c.GetModelSavePVCName(cpodjob)
	pvc := &corev1.PersistentVolumeClaim{}
	if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: modeSavePVCName}, pvc); err != nil {
		if apierrors.IsNotFound(err) {
			logger.Info("modelsave pvc not found, create it")
			volumeMode := corev1.PersistentVolumeFilesystem
			err := c.Client.Create(ctx, &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      modeSavePVCName,
					Namespace: cpodjob.Namespace,
					OwnerReferences: []metav1.OwnerReference{
						c.generateOwnerRefCPodJob(ctx, cpodjob),
					},
				},
				Spec: corev1.PersistentVolumeClaimSpec{
					AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteMany},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceStorage: *resource.NewQuantity((int64(cpodjob.Spec.ModelSaveVolumeSize) * 1024 * 1024), resource.BinarySI),
						},
					},
					StorageClassName: &c.Option.StorageClassName,
					VolumeMode:       &volumeMode,
				},
			})
			if err != nil {
				logger.Error(err, "create modesave pvc failed")
				return nil, err
			}
		}
		return nil, err
	}

	// Check if the PVC is bound
	if pvc.Status.Phase != corev1.ClaimBound {
		logger.Info("ckpt pvc not bound, wait for binding")
		return nil, fmt.Errorf("ckpt pvc not bound, wait for binding")
	}
	return pvc, nil
}

// onOwnerCreateFunc modify creation condition.
func (c *CPodJobReconciler) onOwnerCreateFunc() func(event.CreateEvent) bool {
	return func(e event.CreateEvent) bool {
		cpodjob, ok := e.Object.(*v1beta1.CPodJob)
		if !ok {
			return true
		}
		msg := fmt.Sprintf("Cpodjob %v is created", e.Object.GetName())
		logrus.Info(msg)
		util.CreatedJobsCounterInc(cpodjob.Namespace, string(cpodjob.Spec.JobType))
		util.UpdateJobConditions(&cpodjob.Status, cpodv1beta1.JobCreated, corev1.ConditionTrue, "CpodjobCreated", msg)
		return true
	}
}

// generateOwnerRefCPodJob generates an OwnerReference for a CPodJob object.
// It takes a context.Context and a CPodJob object as input and returns a metav1.OwnerReference.
// The generated OwnerReference contains the APIVersion, Kind, Name, UID, Controller, and BlockOwnerDeletion fields.
func (c *CPodJobReconciler) generateOwnerRefCPodJob(ctx context.Context, cpodjob *v1beta1.CPodJob) metav1.OwnerReference {
	yes := true
	return metav1.OwnerReference{
		APIVersion:         cpodv1beta1.GroupVersion.String(),
		Kind:               "CPodJob",
		Name:               cpodjob.Name,
		UID:                cpodjob.GetUID(),
		Controller:         &yes,
		BlockOwnerDeletion: &yes,
	}
}

func (c *CPodJobReconciler) uploadSavedModel(ctx context.Context, cpodjob *v1beta1.CPodJob, userID, jobName string) error {
	uploadJob := &batchv1.Job{}
	uploadJobName := cpodjob.Name + "-upload"
	completion := int32(1)
	parallelism := int32(1)
	// 拷贝secret
	if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: v1beta1.K8S_SECRET_NAME_FOR_OSS}, &corev1.Secret{}); err != nil {
		if apierrors.IsNotFound(err) {
			publicSecret := &corev1.Secret{}
			if err := c.Client.Get(ctx, client.ObjectKey{Namespace: v1beta1.CPodPublicNamespace, Name: v1beta1.K8S_SECRET_NAME_FOR_OSS}, publicSecret); err != nil {
				if apierrors.IsNotFound(err) {
					return fmt.Errorf("public secret %s not found", v1beta1.K8S_SECRET_NAME_FOR_OSS)
				}
				return fmt.Errorf("failed to get public secret %s: %v", v1beta1.K8S_SECRET_NAME_FOR_OSS, err)
			}
			secret := publicSecret.DeepCopy()
			secret.Namespace = cpodjob.Namespace
			secret.ResourceVersion = ""
			secret.UID = ""
			if err := c.Client.Create(ctx, secret); err != nil {
				return fmt.Errorf("failed to copy secret %s", v1beta1.K8S_SECRET_NAME_FOR_OSS)
			}
		} else {
			return fmt.Errorf("failed to get secret %s", v1beta1.K8S_SECRET_NAME_FOR_OSS)
		}
	}

	// 拷贝 cm
	if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: v1beta1.K8S_CPOD_CM}, &corev1.ConfigMap{}); err != nil {
		if apierrors.IsNotFound(err) {
			publicCm := &corev1.ConfigMap{}
			if err := c.Client.Get(ctx, client.ObjectKey{Namespace: v1beta1.CPodPublicNamespace, Name: v1beta1.K8S_CPOD_CM}, publicCm); err != nil {
				if apierrors.IsNotFound(err) {
					return fmt.Errorf("public configmap %s not found", v1beta1.K8S_CPOD_CM)
				}
				return fmt.Errorf("failed to get public configmap %s: %v", v1beta1.K8S_CPOD_CM, err)
			}
			cm := publicCm.DeepCopy()
			cm.Namespace = cpodjob.Namespace
			cm.ResourceVersion = ""
			cm.UID = ""
			if err := c.Client.Create(ctx, cm); err != nil {
				return fmt.Errorf("failed to copy configmap %s", v1beta1.K8S_CPOD_CM)
			}
		} else {
			return fmt.Errorf("failed to get configmap %s", v1beta1.K8S_CPOD_CM)
		}
	}

	if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: uploadJobName}, uploadJob); err != nil {
		if apierrors.IsNotFound(err) {
			err := c.Client.Create(ctx, &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      uploadJobName,
					Namespace: cpodjob.Namespace,
					OwnerReferences: []metav1.OwnerReference{
						c.generateOwnerRefCPodJob(ctx, cpodjob),
					},
				},
				Spec: batchv1.JobSpec{
					BackoffLimit: &c.Option.ModelUploadJobBackoffLimit,
					Completions:  &completion,
					Parallelism:  &parallelism,
					Template: corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:            "uploadjob",
									Image:           c.Option.ModelUploadJobImage,
									ImagePullPolicy: corev1.PullAlways,
									Command: []string{
										"./modeluploadjob",
										"user-" + userID,
										jobName,
										c.Option.OssOption.BucketName,
									},
									Env: []corev1.EnvVar{
										{
											Name: "access_key",
											ValueFrom: &corev1.EnvVarSource{
												ConfigMapKeyRef: &corev1.ConfigMapKeySelector{
													LocalObjectReference: corev1.LocalObjectReference{
														Name: "cpod-info",
													},
													Key: "access_key",
												},
											},
										},
									},
									EnvFrom: []corev1.EnvFromSource{
										{
											SecretRef: &corev1.SecretEnvSource{
												LocalObjectReference: corev1.LocalObjectReference{
													Name: v1beta1.K8S_SECRET_NAME_FOR_OSS,
												},
											},
										},
									},
									VolumeMounts: []corev1.VolumeMount{
										{
											Name:      "modelsave-pv",
											MountPath: v1beta1.MODELUPLOADER_PVC_MOUNT_PATH,
										},
									},
								},
							},
							Volumes: []corev1.Volume{
								{
									Name: "modelsave-pv",
									VolumeSource: corev1.VolumeSource{
										PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
											ClaimName: c.GetModelSavePVCName(cpodjob),
										},
									},
								},
							},
							RestartPolicy: corev1.RestartPolicyOnFailure,
						},
					},
				},
			})
			if err != nil {
				return err
			}
			util.UpdateJobConditions(&cpodjob.Status, cpodv1beta1.JobModelUploading, corev1.ConditionTrue, "UploadingModel", "modelupload job is running")
		}
		return err
	}
	if uploadJob.Status.Succeeded == 1 {
		util.UpdateJobConditions(&cpodjob.Status, cpodv1beta1.JobModelUploaded, corev1.ConditionTrue, "UploadModelSucceed", "Upload model succeed")
	} else if uploadJob.Status.Failed >= c.Option.ModelUploadJobBackoffLimit {
		util.UpdateJobConditions(&cpodjob.Status, cpodv1beta1.JobModelUploaded, corev1.ConditionFalse, "UploadModelSucceed", "modelupload backofflimit exceed")
	} else {
		util.UpdateJobConditions(&cpodjob.Status, cpodv1beta1.JobModelUploading, corev1.ConditionTrue, "UploadingModel", "modelupload job is running")
		return fmt.Errorf("modelupload job is running")
	}
	return nil
}

// GetModelSavePVCName returns the name of the PVC (Persistent Volume Claim) used to save the model for the given CPodJob.
// The name is generated by appending "-modelsave-pvc" to the name of the CPodJob.
func (c *CPodJobReconciler) GetModelSavePVCName(cpodjob *v1beta1.CPodJob) string {
	return fmt.Sprintf("%s-modelsave-pvc", cpodjob.Name)
}

// If there are two node with label 'feature.node.kubernetes.io/rdma.available=true',the k8s cluster is ib available
func (c *CPodJobReconciler) needAllocateRDMADevice(ctx context.Context, cpodjob *v1beta1.CPodJob) bool {
	nodeList := &corev1.NodeList{}
	if err := c.Client.List(ctx, nodeList, client.MatchingLabels{"feature.node.kubernetes.io/rdma.available": "true"}); err != nil {
		return false
	}
	if len(nodeList.Items) >= 2 && cpodjob.Spec.WorkerReplicas > 1 {
		return true
	}
	return false
}

func (c *CPodJobReconciler) releaseSavedModel(ctx context.Context, cpodjob *v1beta1.CPodJob) error {
	if cpodjob.Spec.ModelSavePath == "" || cpodjob.Spec.ModelSaveVolumeSize == 0 {
		return nil
	}
	// modelSavePVCName := c.GetModelSavePVCName(cpodjob)
	modelSavePVC, err := c.GetModelSavePVC(ctx, cpodjob)
	if err != nil {
		return err
	}

	modelSavePVCCopy := modelSavePVC.DeepCopy()
	needUpdate := false
	OwnerReferences := modelSavePVC.ObjectMeta.GetOwnerReferences()
	for i, ownerReference := range OwnerReferences {
		if ownerReference.UID == cpodjob.UID {
			// Remove the ownerReference from the ownerReferences slice
			OwnerReferences = append(OwnerReferences[:i], OwnerReferences[i+1:]...)
			needUpdate = true
			break
		}
	}
	if needUpdate {
		modelSavePVCCopy.ObjectMeta.SetOwnerReferences(OwnerReferences)
		// Update the modelSavePVC with the modified ownerReferences
		err = c.Client.Update(ctx, modelSavePVCCopy)
		if err != nil {
			return err
		}
	}
	return nil
}

func (c *CPodJobReconciler) generateModelstorage(preTrainModelStoreage cpodv1.ModelStorage, cpodjob *v1beta1.CPodJob) *cpodv1.ModelStorage {
	readableModelstorageName := preTrainModelStoreage.Spec.ModelName + "-" + time.Now().String()
	if cpodjob.Annotations != nil {
		if name, ok := cpodjob.Annotations[v1beta1.CPodModelstorageNameAnno]; ok && name != "" {
			readableModelstorageName = name
		}
	}
	return &cpodv1.ModelStorage{
		ObjectMeta: metav1.ObjectMeta{
			Name:      generateModelstorageName(cpodjob),
			Namespace: cpodjob.Namespace,
			Labels:    cpodjob.Labels,
			Annotations: map[string]string{
				v1beta1.CPodModelstorageBaseNameAnno: preTrainModelStoreage.Spec.ModelName,
			},
		},
		Spec: cpodv1.ModelStorageSpec{
			ModelType: "trained",
			ModelName: readableModelstorageName,
			PVC:       c.GetModelSavePVCName(cpodjob),
			Template:  preTrainModelStoreage.Spec.Template,
		},
	}
}

func (c *CPodJobReconciler) createGeneratedModelstorage(ctx context.Context, cpodjob *v1beta1.CPodJob) error {
	if cpodjob.Spec.ModelSavePath == "" || cpodjob.Spec.ModelSaveVolumeSize == 0 || cpodjob.Spec.PretrainModelName == "" || cpodjob.Spec.PretrainModelPath == "" {
		return nil
	}

	preTrainModelStoreage := cpodv1.ModelStorage{}
	if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: cpodjob.Spec.PretrainModelName}, &preTrainModelStoreage); err != nil {
		return err
	}

	modelstorageName := generateModelstorageName(cpodjob)
	modelstorage := cpodv1.ModelStorage{}

	if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: modelstorageName}, &modelstorage); err != nil {
		if apierrors.IsNotFound(err) {
			if err := c.Client.Create(ctx, c.generateModelstorage(preTrainModelStoreage, cpodjob)); err != nil {
				return err
			}
		}
		return err
	}

	if modelstorage.Status.Phase != "done" {
		modelstorage.Status.Phase = "done"
		modelstorage.Status.Size = preTrainModelStoreage.Status.Size
		return c.Client.Status().Update(ctx, &modelstorage)
	}

	return nil

}

func generateModelstorageName(cpodjob *v1beta1.CPodJob) string {
	modelstorageName := cpodjob.Name + "-modelsavestorage"
	jobName := cpodjob.Name
	if strings.HasSuffix(jobName, "-cpodjob") {
		jobName, _ = strings.CutSuffix(jobName, "-cpodjob")
	}
	if userId, ok := cpodjob.Labels[v1beta1.CPodUserIDLabel]; ok {
		modelstorageName = util.ModelCRDName(fmt.Sprintf(util.OSSUserModelPath, "user-"+userId+"/"+jobName))
	}
	return modelstorageName
}

func (c *CPodJobReconciler) PrepareData(ctx context.Context, cpodjob *v1beta1.CPodJob) error {
	wg := sync.WaitGroup{}
	errChan := make(chan error, 2)
	wg.Add(2)
	go func() {
		defer wg.Done()
		if err := c.prepareModel(ctx, cpodjob); err != nil {
			errChan <- fmt.Errorf("prepare dataset failed: %v", err)
			return
		}
		return
	}()

	go func() {
		defer wg.Done()
		if err := c.prepareDataset(ctx, cpodjob); err != nil {
			errChan <- fmt.Errorf("prepare dataset failed: %v", err)
			return
		}
		return
	}()
	wg.Wait()
	if len(errChan) != 0 {
		for err := range errChan {
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func (c *CPodJobReconciler) prepareModel(ctx context.Context, cpodjob *v1beta1.CPodJob) error {
	logrus.Info("DEBUG ", "cpodjob", cpodjob.Spec, "Annotations", cpodjob.Annotations)
	if cpodjob.Spec.PretrainModelName == "" {
		return nil
	}
	modelSize := int64(0)
	modelReadableName := ""
	modelTemplate := ""
	if cpodjob.Annotations != nil {
		if sizeStr, ok := cpodjob.Annotations[v1beta1.CPodPreTrainModelSizeAnno]; ok {
			size, err := strconv.ParseInt(sizeStr, 10, 64)
			if err != nil {
				return fmt.Errorf("failed to parse model size %v: %v", sizeStr, err)
			}
			modelSize = size
		}
		if name, ok := cpodjob.Annotations[v1beta1.CPodPreTrainModelReadableNameAnno]; ok {
			modelReadableName = name
		}

		if template, ok := cpodjob.Annotations[v1beta1.CPodPreTrainModelTemplateAnno]; ok {
			modelTemplate = template
		}
	}
	if cpodjob.Spec.PretrainModelIsPublic {
		modelName := cpodjob.Spec.PretrainModelName + v1beta1.CPodPublicStorageSuffix
		ms := &cpodv1.ModelStorage{}
		if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: modelName}, ms); err != nil {
			if apierrors.IsNotFound(err) {
				publicMs := &cpodv1.ModelStorage{}
				if err := c.Client.Get(ctx, client.ObjectKey{Namespace: v1beta1.CPodPublicNamespace, Name: cpodjob.Spec.PretrainModelName}, publicMs); err != nil {
					if apierrors.IsNotFound(err) {
						if createdMs, err := createModelstorage(ctx, c.Client, cpodjob.Spec.PretrainModelName, modelReadableName, modelSize, modelTemplate, v1beta1.CPodPublicNamespace, c.Option.StorageClassName); err != nil {
							return fmt.Errorf("failed to create model storage for public model %s: %v", cpodjob.Spec.PretrainModelName, err)
						} else {
							publicMs = createdMs
						}
					} else {
						return fmt.Errorf("failed to get public model %s: %v", cpodjob.Spec.PretrainModelName, err)
					}
				}
				if publicMs != nil && publicMs.Status.Phase != "done" {
					jobName := "model-" + cpodjob.Spec.PretrainModelName
					job := &batchv1.Job{}
					if err := c.Client.Get(ctx, client.ObjectKey{Namespace: v1beta1.CPodPublicNamespace, Name: jobName}, job); err != nil {
						if apierrors.IsNotFound(err) {
							if err := CreateDownloadJob(ctx, c.Client, c.Option.OssOption, "model", cpodjob.Spec.PretrainModelName, modelReadableName, modelSize, cpodjob.Namespace, v1beta1.CPodPublicNamespace); err != nil {
								return fmt.Errorf("failed to create download job for public model %s: %v", cpodjob.Spec.PretrainModelName, err)
							}
						} else {
							return fmt.Errorf("failed to get public model %s: %v", cpodjob.Spec.PretrainModelName, err)
						}
					}
					if job.Status.Succeeded != 1 {
						return fmt.Errorf("public model downloader job %s is running: %v", jobName, job.Status.Succeeded)
					}
					return fmt.Errorf("public model %s is not done", cpodjob.Spec.PretrainModelName)
				}
				if err := CopyPublicModelStorage(ctx, c.Client, cpodjob.Spec.PretrainModelName, cpodjob.Namespace); err != nil {
					return fmt.Errorf("failed to copy public model %s: %v", cpodjob.Spec.PretrainModelName, err)
				}
				return nil
			} else {
				return fmt.Errorf("failed to get public model %v's copy  %s: %v", cpodjob.Spec.PretrainModelName, modelName, err)
			}
		}
		if ms.Status.Phase != "done" {
			return fmt.Errorf("public model copy  %s is not done", cpodjob.Spec.PretrainModelName)
		}
		return nil
	}
	ms := &cpodv1.ModelStorage{}
	if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: cpodjob.Spec.PretrainModelName}, ms); err != nil {
		if apierrors.IsNotFound(err) {
			if createdMs, err := createModelstorage(ctx, c.Client, cpodjob.Spec.PretrainModelName, modelReadableName, modelSize, modelTemplate, cpodjob.Namespace, c.Option.StorageClassName); err != nil {
				return fmt.Errorf("failed to create model storage for private model %s: %v", cpodjob.Spec.PretrainModelName, err)
			} else {
				ms = createdMs
			}
		} else {
			return fmt.Errorf("failed to get private model %s: %v", cpodjob.Spec.PretrainModelName, err)
		}
	}
	if ms != nil && ms.Status.Phase != "done" {
		jobName := "model-" + cpodjob.Spec.PretrainModelName
		job := &batchv1.Job{}
		if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: jobName}, job); err != nil {
			if apierrors.IsNotFound(err) {
				if err := CreateDownloadJob(ctx, c.Client, c.Option.OssOption, "model", cpodjob.Spec.PretrainModelName, modelReadableName, modelSize, cpodjob.Namespace, cpodjob.Namespace); err != nil {
					return fmt.Errorf("failed to create download job for private model %s: %v", cpodjob.Spec.PretrainModelName, err)
				}
			} else {
				return fmt.Errorf("failed to get private model %s: %v", cpodjob.Spec.PretrainModelName, err)
			}
		}
		if job.Status.Succeeded != 1 {
			return fmt.Errorf("model downloader job %s is running: %v", jobName, job.Status.Succeeded)
		}
		return fmt.Errorf("private model %s is not done", cpodjob.Spec.PretrainModelName)
	}
	return nil
}

func (c *CPodJobReconciler) prepareDataset(ctx context.Context, cpodjob *v1beta1.CPodJob) error {
	logger := log.FromContext(ctx)
	if cpodjob.Spec.DatasetName == "" {
		return nil
	}
	datasetSize := int64(0)
	datasetReadableName := ""
	if cpodjob.Annotations != nil {
		if sizeStr, ok := cpodjob.Annotations[v1beta1.CPodDatasetSizeAnno]; ok {
			size, err := strconv.ParseInt(sizeStr, 10, 64)
			if err != nil {
				return fmt.Errorf("failed to parse datasize size %v: %v", sizeStr, err)
			}
			datasetSize = size
		}
		if name, ok := cpodjob.Annotations[v1beta1.CPodDatasetlReadableNameAnno]; ok {
			datasetReadableName = name
		}
	}
	if cpodjob.Spec.DatasetIsPublic {
		dsName := cpodjob.Spec.DatasetName + v1beta1.CPodPublicStorageSuffix
		ds := &cpodv1.DataSetStorage{}
		if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: dsName}, ds); err != nil {
			if apierrors.IsNotFound(err) {
				logger.Info("public dataset copy not found, create it", "dataset", cpodjob.Spec.DatasetName)
				publicDs := &cpodv1.DataSetStorage{}
				if err := c.Client.Get(ctx, client.ObjectKey{Namespace: v1beta1.CPodPublicNamespace, Name: cpodjob.Spec.DatasetName}, publicDs); err != nil {
					if apierrors.IsNotFound(err) {
						if createdDs, err := createDatasetStorage(ctx, c.Client, cpodjob.Spec.DatasetName, datasetReadableName, datasetSize, v1beta1.CPodPublicNamespace, c.Option.StorageClassName); err != nil {
							return fmt.Errorf("failed to create dataset storage for public model %s: %v", cpodjob.Spec.DatasetName, err)
						} else {
							publicDs = createdDs
						}
					} else {
						return fmt.Errorf("failed to get public dataset %s: %v", cpodjob.Spec.DatasetName, err)
					}
				}
				if publicDs != nil && publicDs.Status.Phase != "done" {
					jobName := "dataset-" + cpodjob.Spec.DatasetName
					job := &batchv1.Job{}
					if err := c.Client.Get(ctx, client.ObjectKey{Namespace: v1beta1.CPodPublicNamespace, Name: jobName}, job); err != nil {
						if apierrors.IsNotFound(err) {
							if err := CreateDownloadJob(ctx, c.Client, c.Option.OssOption, "dataset", cpodjob.Spec.DatasetName, datasetReadableName, datasetSize, cpodjob.Namespace, v1beta1.CPodPublicNamespace); err != nil {
								return fmt.Errorf("failed to create download job for public dataset %s: %v", cpodjob.Spec.DatasetName, err)
							}
						} else {
							return fmt.Errorf("failed to get public dataset %s: %v", cpodjob.Spec.DatasetName, err)
						}
					}
					if job.Status.Succeeded != 1 {
						return fmt.Errorf("public dataset downloader job %s is running: %v", jobName, job.Status.Succeeded)
					}
					return fmt.Errorf("public dataset %s is not done", cpodjob.Spec.DatasetName)
				}
				if err := CopyPublicDatasetStorage(ctx, c.Client, cpodjob.Spec.DatasetName, cpodjob.Namespace); err != nil {
					return fmt.Errorf("failed to copy public model %s: %v", cpodjob.Spec.PretrainModelName, err)
				}
				return nil
			} else {
				return fmt.Errorf("failed to get public dataset %v's copy  %s: %v", cpodjob.Spec.DatasetName, dsName, err)
			}
		}
		if ds.Status.Phase != "done" {
			return fmt.Errorf("public dataset copy  %s is not done", cpodjob.Spec.DatasetName)
		}
		return nil
	}
	ds := &cpodv1.DataSetStorage{}
	if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: cpodjob.Spec.DatasetName}, ds); err != nil {
		if apierrors.IsNotFound(err) {
			if createdDs, err := createDatasetStorage(ctx, c.Client, cpodjob.Spec.DatasetName, datasetReadableName, datasetSize, cpodjob.Namespace, c.Option.StorageClassName); err != nil {
				return fmt.Errorf("failed to create dataset storage for private dataset %s: %v", cpodjob.Spec.DatasetName, err)
			} else {
				ds = createdDs
			}
		} else {
			return fmt.Errorf("failed to get private dataset %s: %v", cpodjob.Spec.DatasetName, err)
		}
	}
	if ds != nil && ds.Status.Phase != "done" {
		jobName := "dataset-" + cpodjob.Spec.DatasetName
		job := &batchv1.Job{}
		if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: jobName}, job); err != nil {
			if apierrors.IsNotFound(err) {
				if err := CreateDownloadJob(ctx, c.Client, c.Option.OssOption, "dataset", cpodjob.Spec.DatasetName, datasetReadableName, datasetSize, cpodjob.Namespace, cpodjob.Namespace); err != nil {
					return fmt.Errorf("failed to create download job for private dataset %s: %v", cpodjob.Spec.DatasetName, err)
				}
			} else {
				return fmt.Errorf("failed to get private dataset %s: %v", cpodjob.Spec.DatasetName, err)
			}
		}
		if job.Status.Succeeded != 1 {
			return fmt.Errorf("dataset downloader job %s is running: %v", jobName, job.Status.Succeeded)
		}
		return fmt.Errorf("private dataset %s is not done", cpodjob.Spec.DatasetName)
	}
	return nil
}

func createModelstorage(ctx context.Context, kubeclient client.Client, dataID, dataName string, dataSize int64, template, namespace string, storageClassName string) (*cpodv1.ModelStorage, error) {
	ossPath := util.ResourceToOSSPath("model", dataName)

	pvcName := util.ModelPVCName(ossPath)
	pvcSize := fmt.Sprintf("%dMi", dataSize*12/10/1024/1024)
	if err := kubeclient.Get(ctx, client.ObjectKey{Namespace: namespace, Name: pvcName}, &corev1.PersistentVolumeClaim{}); err != nil {
		if apierrors.IsNotFound(err) {
			err := kubeclient.Create(ctx, &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      pvcName,
					Namespace: namespace,
				},
				Spec: corev1.PersistentVolumeClaimSpec{
					AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteMany},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceStorage: resource.MustParse(pvcSize),
						},
					},
					StorageClassName: &storageClassName,
				},
			})
			if err != nil && !apierrors.IsAlreadyExists(err) {
				return nil, fmt.Errorf("failed to create pvc %s, : %v, pvcSize %v, datasize %v ", pvcName, err, pvcSize, dataSize)
			}
		} else {
			return nil, fmt.Errorf("failed to get pvc %s: %v", pvcName, err)
		}
	}

	modelstorage := &cpodv1.ModelStorage{}
	if err := kubeclient.Get(ctx, client.ObjectKey{Namespace: namespace, Name: dataID}, modelstorage); err != nil {
		if apierrors.IsNotFound(err) {
			modelstorage = &cpodv1.ModelStorage{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dataID,
					Namespace: namespace,
				},
				Spec: cpodv1.ModelStorageSpec{
					ModelType: "oss",
					ModelName: dataName,
					PVC:       pvcName,
					Template:  template,
				},
			}
			if err := kubeclient.Create(ctx, modelstorage); err != nil && !apierrors.IsAlreadyExists(err) {
				return nil, fmt.Errorf("failed to create modelstorage %s: %v", dataID, err)
			}
		} else {
			return nil, fmt.Errorf("failed to get modelstorage %s: %v", dataID, err)
		}
	}

	return modelstorage, nil
}

func createDatasetStorage(ctx context.Context, kubeclient client.Client, dataID, dataName string, dataSize int64, namespace string, storageClassName string) (*cpodv1.DataSetStorage, error) {
	ossPath := util.ResourceToOSSPath("dataset", dataName)

	pvcName := util.DatasetPVCName(ossPath)
	pvcSize := fmt.Sprintf("%dMi", dataSize*12/10/1024/1024)
	if err := kubeclient.Get(ctx, client.ObjectKey{Namespace: namespace, Name: pvcName}, &corev1.PersistentVolumeClaim{}); err != nil {
		if apierrors.IsNotFound(err) {
			err := kubeclient.Create(ctx, &corev1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      pvcName,
					Namespace: namespace,
				},
				Spec: corev1.PersistentVolumeClaimSpec{
					AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteMany},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceStorage: resource.MustParse(pvcSize),
						},
					},
					StorageClassName: &storageClassName,
				},
			})
			if err != nil && !apierrors.IsAlreadyExists(err) {
				return nil, fmt.Errorf("failed to create pvc %s: %v", pvcName, err)
			}
		} else {
			return nil, fmt.Errorf("failed to get pvc %s: %v", pvcName, err)
		}
	}

	datasetstorage := &cpodv1.DataSetStorage{}
	if err := kubeclient.Get(ctx, client.ObjectKey{Namespace: namespace, Name: dataID}, datasetstorage); err != nil {
		if apierrors.IsNotFound(err) {
			datasetstorage = &cpodv1.DataSetStorage{
				ObjectMeta: metav1.ObjectMeta{
					Name:      dataID,
					Namespace: namespace,
				},
				Spec: cpodv1.DataSetStorageSpec{
					DatasetType: "oss",
					DatasetName: dataName,
					PVC:         pvcName,
				},
			}
			if err := kubeclient.Create(ctx, datasetstorage); err != nil && !apierrors.IsAlreadyExists(err) {
				return nil, fmt.Errorf("failed to create datasetstorage %s: %v", dataID, err)
			}
		} else {
			return nil, fmt.Errorf("failed to get datasetstorage %s: %v", dataID, err)
		}
	}

	return datasetstorage, nil
}

// dataType: model、dataset
func CreateDownloadJob(ctx context.Context, kubeclient client.Client, OssOption OssOption, dataType string, dataID, dataName string, dataSize int64, userId string, namespace string) error {
	ossPath := util.ResourceToOSSPath(dataType, dataName)

	var pvcName string
	if dataType == "model" {
		pvcName = util.ModelPVCName(ossPath)
	} else {
		pvcName = util.DatasetPVCName(ossPath)
	}

	// 构造oss路径
	completionMode := batchv1.NonIndexedCompletion

	// type-dataName
	downloadJobName := fmt.Sprintf("%s-%s", dataType, dataID)
	job := batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      downloadJobName,
			Labels:    map[string]string{v1beta1.CPodJobSourceLabel: v1beta1.CPodJobSource},
		},
		Spec: batchv1.JobSpec{
			Parallelism:  ptr.Int32(1),
			Completions:  ptr.Int32(1),
			BackoffLimit: ptr.Int32(6),
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Volumes: []corev1.Volume{
						{
							Name: "data-volume",
							VolumeSource: corev1.VolumeSource{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
								ClaimName: pvcName,
								ReadOnly:  false,
							}},
						},
					},
					Containers: []corev1.Container{
						{
							Name:  "downloader",
							Image: OssOption.DownloaderImage,
							Args: []string{
								"-g",
								"cpod.cpod",
								"-v",
								"v1",
								"-p",
								dataType + "storages",
								"-n",
								namespace,
								"--name",
								dataID,
								"oss",
								fmt.Sprintf("oss://%v/%v", OssOption.BucketName, ossPath),
								"-t",
								fmt.Sprintf("%d", dataSize),
								"--endpoint",
								"https://oss-cn-beijing.aliyuncs.com",
								"--access_id",
								OssOption.OssAK,
								"--access_key",
								OssOption.OssAS,
							},
							VolumeMounts: []corev1.VolumeMount{
								{
									MountPath: "/data",
									Name:      "data-volume",
								},
							},
							ImagePullPolicy: corev1.PullIfNotPresent,
						},
					},
					RestartPolicy:                 "Never",
					TerminationGracePeriodSeconds: ptr.Int64(30),
					DNSPolicy:                     "ClusterFirst",
					ImagePullSecrets: []corev1.LocalObjectReference{
						{Name: "aliyun-enterprise-registry"},
					},
				},
			},
			CompletionMode: &completionMode,
		},
	}
	return kubeclient.Create(ctx, &job)
}
