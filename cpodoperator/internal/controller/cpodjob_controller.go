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

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
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

type CPodJobOption struct {
	StorageClassName           string
	ModelUploadJobImage        string
	ModelUploadJobBackoffLimit int32
	ModelUploadOssBucketName   string
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

	if util.IsSucceeded(cpodjob.Status) && cpodjob.Spec.UploadModel && cpodjob.Spec.ModelSavePath != "" {
		err := c.uploadSavedModel(ctx, cpodjob)
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

	if cpodjob.Spec.CKPTPath != "" {
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
		dataset := &cpodv1.DataSetStorage{}
		if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: cpodjob.Spec.DatasetName}, dataset); err != nil {
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
		pretrainModel := &cpodv1.ModelStorage{}
		if err := c.Client.Get(ctx, client.ObjectKey{Namespace: cpodjob.Namespace, Name: cpodjob.Spec.PretrainModelName}, pretrainModel); err != nil {
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
			err := c.Client.Create(ctx, &corev1.PersistentVolumeClaim{
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
							corev1.ResourceStorage: *resource.NewQuantity(int64(cpodjob.Spec.CKPTVolumeSize*1024*1024), resource.BinarySI),
						},
					},
					StorageClassName: &c.Option.StorageClassName,
					VolumeMode:       &volumeMode,
				},
			})
			if err != nil {
				logger.Error(err, "create ckpt pvc failed")
				return nil, err
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

func (c *CPodJobReconciler) uploadSavedModel(ctx context.Context, cpodjob *v1beta1.CPodJob) error {
	uploadJob := &batchv1.Job{}
	uploadJobName := cpodjob.Name + "-upload"
	completion := int32(1)
	parallelism := int32(1)
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
										cpodjob.Name,
										c.Option.ModelUploadOssBucketName,
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
