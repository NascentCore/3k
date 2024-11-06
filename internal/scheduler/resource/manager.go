package resource

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path"
	"strings"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/orm"
	"sxwl/3k/pkg/storage"
	"time"

	"github.com/Masterminds/squirrel"
	"github.com/pkg/errors"
	"github.com/zeromicro/go-zero/core/logx"
)

type Manager struct {
	ctx context.Context
	logx.Logger
	svcCtx *svc.ServiceContext
}

func NewManager(svcCtx *svc.ServiceContext) *Manager {
	ctx := context.Background()
	return &Manager{
		ctx:    ctx,
		Logger: logx.WithContext(ctx),
		svcCtx: svcCtx,
	}
}

func (m *Manager) SyncOSS() {
	m.syncOSSModels()
	m.syncOSSDatasets()
	m.syncOSSAdapter()
}

func (m *Manager) syncOSSModels() {
	OssResourceModel := m.svcCtx.OssResourceModel

	// resource in db
	resourcesInDB, err := OssResourceModel.FindAll(m.ctx, "")
	if err != nil && !errors.Is(err, model.ErrNotFound) {
		m.Errorf("OssResourceModel FindAll err: %v", err)
		return
	}

	resourceMap := make(map[string]bool)
	for _, resource := range resourcesInDB {
		resourceMap[resource.ResourceId] = true
	}

	var modelsToInsert, modelsToUpdate []model.SysOssResource

	// public models
	// ListDir 只能查询1000个匹配前缀的文件，小批量数据ok，更完善还是需要有db来存储模型元数据
	dirs, err := storage.ListDir(m.svcCtx.Config.OSS.Bucket, m.svcCtx.Config.OSS.PublicModelDir, 2)
	if err != nil {
		m.Errorf("SyncOSS err: %v", err)
		return
	}

	for dir, size := range dirs {
		canFinetune, _, err := storage.ExistFile(m.svcCtx.Config.OSS.Bucket,
			path.Join(dir, m.svcCtx.Config.OSS.FinetuneTagFile))
		if err != nil {
			m.Errorf("SyncOSS ExistFile err: %v", err)
			return
		}
		canInference, _, err := storage.ExistFile(m.svcCtx.Config.OSS.Bucket,
			path.Join(dir, m.svcCtx.Config.OSS.InferenceTagFile))
		if err != nil {
			m.Errorf("SyncOSS ExistFile err: %v", err)
			return
		}

		modelName := strings.TrimPrefix(strings.TrimSuffix(dir, "/"), m.svcCtx.Config.OSS.PublicModelDir)

		meta := model.OssResourceModelMeta{
			Template:     storage.ModelTemplate(m.svcCtx.Config.OSS.Bucket, modelName),
			Category:     consts.ModelCategoryChat, // oss上没有这个信息，默认就当做chat模型
			CanFinetune:  canFinetune,
			CanInference: canInference,
		}
		metaJson, err := json.Marshal(meta)
		if err != nil {
			m.Errorf("SyncOSS meta marshal err: %v", err)
			return
		}

		resourceID := storage.ModelCRDName(storage.ResourceToOSSPath(consts.Model, modelName))
		resource := model.SysOssResource{
			ResourceId:   resourceID,
			ResourceType: "model",
			ResourceName: modelName,
			ResourceSize: size,
			Public:       model.CachePublic,
			UserId:       "public",
			Meta:         string(metaJson),
		}

		_, ok := resourceMap[resourceID]
		if ok {
			modelsToUpdate = append(modelsToUpdate, resource)
		} else {
			modelsToInsert = append(modelsToInsert, resource)
		}
	}

	// insert
	err = m.OSSInsert(modelsToInsert)
	if err != nil {
		m.Errorf("SyncOSS insert err: %v", err)
		return
	}

	// update
	err = m.OSSUpdate(modelsToUpdate)
	if err != nil {
		m.Errorf("SyncOSS update err: %v", err)
		return
	}
}

func (m *Manager) syncOSSDatasets() {
	OssResourceModel := m.svcCtx.OssResourceModel

	// resource in db
	resourcesInDB, err := OssResourceModel.FindAll(m.ctx, "")
	if err != nil && !errors.Is(err, model.ErrNotFound) {
		m.Errorf("OssResourceModel FindAll err: %v", err)
		return
	}

	resourceMap := make(map[string]bool)
	for _, resource := range resourcesInDB {
		resourceMap[resource.ResourceId] = true
	}

	var datasetsToInsert, datasetsToUpdate []model.SysOssResource

	// public datasets
	// ListDir 只能查询1000个匹配前缀的文件，小批量数据ok，更完善还是需要有db来存储模型元数据
	dirs, err := storage.ListDir(m.svcCtx.Config.OSS.Bucket, m.svcCtx.Config.OSS.PublicDatasetDir, 2)
	if err != nil {
		m.Errorf("SyncOSS err: %v", err)
		return
	}

	for dir, size := range dirs {
		datasetName := strings.TrimPrefix(strings.TrimSuffix(dir, "/"), m.svcCtx.Config.OSS.PublicDatasetDir)

		resourceID := storage.DatasetCRDName(storage.ResourceToOSSPath(consts.Dataset, datasetName))
		resource := model.SysOssResource{
			ResourceId:   resourceID,
			ResourceType: consts.Dataset,
			ResourceName: datasetName,
			ResourceSize: size,
			Public:       model.CachePublic,
			UserId:       "public",
			Meta:         "{}",
		}

		_, ok := resourceMap[resourceID]
		if ok {
			datasetsToUpdate = append(datasetsToUpdate, resource)
		} else {
			datasetsToInsert = append(datasetsToInsert, resource)
		}
	}

	// insert
	err = m.OSSInsert(datasetsToInsert)
	if err != nil {
		m.Errorf("SyncOSS insert err: %v", err)
		return
	}

	// update
	err = m.OSSUpdate(datasetsToUpdate)
	if err != nil {
		m.Errorf("SyncOSS update err: %v", err)
		return
	}
}

func (m *Manager) syncOSSAdapter() {
	OssResourceModel := m.svcCtx.OssResourceModel

	// resource in db
	resourcesInDB, err := OssResourceModel.FindAll(m.ctx, "")
	if err != nil && !errors.Is(err, model.ErrNotFound) {
		m.Errorf("OssResourceModel FindAll err: %v", err)
		return
	}

	resourceMap := make(map[string]bool)
	for _, resource := range resourcesInDB {
		resourceMap[resource.ResourceId] = true
	}

	var adaptersToInsert, adaptersToUpdate []model.SysOssResource

	// public adapters
	// ListDir 只能查询1000个匹配前缀的文件，小批量数据ok，更完善还是需要有db来存储模型元数据
	dirs, err := storage.ListDir(m.svcCtx.Config.OSS.Bucket, m.svcCtx.Config.OSS.PublicAdapterDir, 2)
	if err != nil {
		m.Errorf("SyncOSS err: %v", err)
		return
	}

	for dir, size := range dirs {
		adapterName := strings.TrimPrefix(strings.TrimSuffix(dir, "/"), m.svcCtx.Config.OSS.PublicAdapterDir)

		meta := model.OssResourceAdapterMeta{
			BaseModel: "default",
		}
		metaJson, err := json.Marshal(meta)
		if err != nil {
			m.Errorf("SyncOSS meta marshal err: %v", err)
			return
		}

		resourceID := storage.DatasetCRDName(storage.ResourceToOSSPath(consts.Adapter, adapterName))
		resource := model.SysOssResource{
			ResourceId:   resourceID,
			ResourceType: consts.Adapter,
			ResourceName: adapterName,
			ResourceSize: size,
			Public:       model.CachePublic,
			UserId:       "public",
			Meta:         string(metaJson),
		}

		_, ok := resourceMap[resourceID]
		if ok {
			adaptersToUpdate = append(adaptersToUpdate, resource)
		} else {
			adaptersToInsert = append(adaptersToInsert, resource)
		}
	}

	// insert
	err = m.OSSInsert(adaptersToInsert)
	if err != nil {
		m.Errorf("SyncOSS insert err: %v", err)
		return
	}

	// update
	err = m.OSSUpdate(adaptersToUpdate)
	if err != nil {
		m.Errorf("SyncOSS update err: %v", err)
		return
	}
}

func (m *Manager) OSSInsert(resources []model.SysOssResource) error {
	if len(resources) == 0 {
		return nil
	}

	insertBuilder := m.svcCtx.OssResourceModel.InsertBuilder().Columns(
		"resource_id",
		"resource_type",
		"resource_name",
		"resource_size",
		"public",
		"user_id",
		"meta",
	)
	for _, resource := range resources {
		insertBuilder = insertBuilder.Values(
			resource.ResourceId,
			resource.ResourceType,
			resource.ResourceName,
			resource.ResourceSize,
			resource.Public,
			resource.UserId,
			resource.Meta,
		)
	}

	query, args, err := insertBuilder.ToSql()
	if err != nil {
		m.Errorf("insertBuilder.ToSql err=%s", err)
		return err
	}

	_, err = m.svcCtx.DB.Exec(query, args...)
	if err != nil {
		m.Errorf("insert exec err=%s", err)
		return err
	}

	return nil
}

func (m *Manager) OSSUpdate(resources []model.SysOssResource) error {
	for _, resource := range resources {
		updateBuilder := m.svcCtx.OssResourceModel.UpdateBuilder().Set(
			"resource_type", resource.ResourceType,
		).Set(
			"resource_name", resource.ResourceName,
		).Set(
			"resource_size", resource.ResourceSize,
		).Set(
			"public", resource.Public,
		).Set(
			"user_id", resource.UserId,
		).Set(
			"meta", resource.Meta,
		).Set(
			"updated_at", time.Now(),
		).Where(
			"resource_id = ?", resource.ResourceId,
		)

		query, args, err := updateBuilder.ToSql()
		if err != nil {
			m.Errorf("updateBuilder.ToSql err=%s", err)
			return err
		}

		_, err = m.svcCtx.DB.Exec(query, args...)
		if err != nil {
			m.Errorf("update exec err=%s", err)
			return err
		}
	}

	return nil
}

// HFLoad 处理huggingface资源同步任务
func (m *Manager) HFLoad(task *model.ResourceSyncTask) error {
	taskModel := m.svcCtx.ResourceSyncTaskModel
	// 1. 更新任务状态为获取meta信息中
	task.Status = model.ResourceSyncTaskStatusGettingMeta
	if err := taskModel.Update(context.Background(), task); err != nil {
		return fmt.Errorf("update task status failed: %v", err)
	}

	// 构建HF API请求URL
	var apiUrl string
	switch task.ResourceType {
	case consts.Model:
		apiUrl = fmt.Sprintf("https://huggingface.co/api/models/%s", task.ResourceId)
	case consts.Dataset:
		apiUrl = fmt.Sprintf("https://huggingface.co/api/datasets/%s", task.ResourceId)
	default:
		return fmt.Errorf("unsupported resource type: %s", task.ResourceType)
	}

	// 创建HTTP请求
	req, err := http.NewRequest(http.MethodGet, apiUrl, nil)
	if err != nil {
		return fmt.Errorf("create request failed: %v", err)
	}

	// 发送请求
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %v", err)
	}
	defer resp.Body.Close()

	// 检查响应状态码
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("request failed with status: %d", resp.StatusCode)
	}

	// 读取响应内容
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read response failed: %v", err)
	}

	// 将响应内容设置为任务的meta信息
	task.Meta = orm.NullString(string(body))

	// 读取README
	readmeUrl := fmt.Sprintf("https://huggingface.co/%s/raw/main/README.md", task.ResourceId)
	readmeReq, err := http.NewRequest(http.MethodGet, readmeUrl, nil)
	if err != nil {
		return fmt.Errorf("create readme request failed: %v", err)
	}

	readmeResp, err := client.Do(readmeReq)
	if err != nil {
		return fmt.Errorf("readme request failed: %v", err)
	}
	defer readmeResp.Body.Close()

	// 如果返回404则说明没有README
	if readmeResp.StatusCode == http.StatusNotFound {
		task.Readme = orm.NullString("")
	} else if readmeResp.StatusCode == http.StatusOK {
		readmeBody, err := io.ReadAll(readmeResp.Body)
		if err != nil {
			return fmt.Errorf("read readme response failed: %v", err)
		}
		task.Readme = orm.NullString(string(readmeBody))
	} else {
		return fmt.Errorf("readme request failed with status: %d", readmeResp.StatusCode)
	}

	// 2. 更新任务状态为获取meta信息完成
	task.Status = model.ResourceSyncTaskStatusGettingMetaDone
	if err := taskModel.Update(context.Background(), task); err != nil {
		return fmt.Errorf("update task status failed: %v", err)
	}

	return nil
}

// StartLoadTask 启动资源同步任务处理
func (m *Manager) StartLoadTask() {
	taskModel := m.svcCtx.ResourceSyncTaskModel
	ossResourceModel := m.svcCtx.OssResourceModel

	for {
		// 1. 查询pending状态的任务
		builder := taskModel.AllFieldsBuilder().
			Where("status IN (?, ?)", model.ResourceSyncTaskStatusPending, model.ResourceSyncTaskStatusUploaded)

		task, err := taskModel.FindOneByQuery(context.Background(), builder)
		if err == model.ErrNotFound {
			// 没有待处理的任务,等待一段时间后继续
			time.Sleep(10 * time.Second)
			continue
		}
		if err != nil {
			m.Errorf("query pending task failed: %v", err)
			continue
		}

		if task.Status == model.ResourceSyncTaskStatusUploaded {
			// 上传完成的任务直接记录资源表
			var resourceID string
			if task.ResourceType == consts.Model {
				resourceID = storage.ModelCRDName(storage.ResourceToOSSPath(consts.Model, task.ResourceId))
			} else if task.ResourceType == consts.Dataset {
				resourceID = storage.DatasetCRDName(storage.ResourceToOSSPath(consts.Dataset, task.ResourceId))
			} else {
				m.Errorf("unsupported resource type: %s", task.ResourceType)
				continue
			}

			meta := model.OssResourceModelMeta{
				Template:     "default",
				Category:     consts.ModelCategoryChat, // oss上没有这个信息，默认就当做chat模型
				CanFinetune:  true,
				CanInference: true,
			}
			metaJson, err := json.Marshal(meta)
			if err != nil {
				m.Errorf("marshal meta failed: %v", err)
				continue
			}
			if _, err := ossResourceModel.Insert(context.Background(), &model.SysOssResource{
				ResourceId:   resourceID,
				ResourceType: task.ResourceType,
				ResourceName: task.ResourceId,
				ResourceSize: task.Size,
				Public:       model.CachePublic,
				UserId:       "public",
				Meta:         string(metaJson),
				Readme:       task.Readme,
			}); err != nil {
				m.Errorf("record resource failed: %v", err)
				continue
			}

			// 更新任务状态为记录资源表完成
			_, err = taskModel.UpdateColsByCond(context.Background(), taskModel.UpdateBuilder().Where(squirrel.Eq{
				"id": task.Id,
			}).SetMap(map[string]interface{}{
				"status": model.ResourceSyncTaskStatusRecord,
			}))
			if err != nil {
				m.Errorf("update task status failed: %v", err)
				continue
			}
		} else {
			// 新创建的下载任务根据source选择处理方式
			switch task.Source {
			case "huggingface":
				if err := m.HFLoad(task); err != nil {
					// 更新任务状态为失败
					task.Status = model.ResourceSyncTaskStatusFailed
					task.ErrInfo = err.Error()
					_ = taskModel.Update(context.Background(), task)
					continue
				}
			default:
				// 不支持的source,更新任务状态为失败
				task.Status = model.ResourceSyncTaskStatusFailed
				task.ErrInfo = "unsupported source"
				_ = taskModel.Update(context.Background(), task)
				continue
			}
		}
	}
}
