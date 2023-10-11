// NO_TEST_NEEDED
package db

import (
	"github.com/golang/glog"
	"gorm.io/gorm"
)

type CpodResources struct {
	Cpods []CpodResource `json:"cpods"`
}

type CpodResource struct {
	gorm.Model
	Id       int     `json:"id,omitempty"`
	GroupId  string  `json:"group_id,omitempty"`
	CpodId   string  `json:"cpod_id"`
	GpuTotal float32 `json:"gpu_total"`
	GpuUsed  float32 `json:"gpu_used"`
	GpuFree  float32 `json:"gpu_free"`
}

func (r *CpodResource) CreateResource(db *gorm.DB) (string, error) {
	results := db.Table("cpod_resource").Where("cpod_id =?", r.CpodId)
	if results.Error != nil {
		if results.Error == gorm.ErrRecordNotFound {
			results := db.Table("cpod_resource").Create(r)
			if results.Error != nil {
				glog.Fatalf("create cpod resource error: %s", results.Error.Error())
				return results.Error.Error(), results.Error
			}
		}
	} else {
		results := db.Table("cpod_resource").Updates(r)
		if results.Error != nil {
			glog.Fatalf("update cpod resource error: %s", results.Error.Error())
			return results.Error.Error(), results.Error
		}
	}
	return "", nil
}

func (r *CpodResource) UpdateResource(db *gorm.DB) (string, error) {
	results := db.Table("cpod_resource").Updates(r)
	if results.Error != nil {
		glog.Errorf("update cpod resource error: %s", results.Error.Error())
		return results.Error.Error(), results.Error
	}
	return "", nil
}

func (r *CpodResource) SelectResource(db *gorm.DB) (*CpodResources, error) {
	var cpodResources CpodResources
	results := db.Table("cpod_resource").Find(r)
	if results.Error != nil {
		if results.Error == gorm.ErrRecordNotFound {
			glog.Errorf("select cpod resource error: %s", results.Error.Error())
			return nil, results.Error
		}
	}
	results.Scan(&cpodResources.Cpods)
	glog.V(5).Infof("selected cpod resource: %v", cpodResources.Cpods)
	return &cpodResources, nil
}

func (r *CpodResource) DeleteResource(db *gorm.DB) (string, error) {
	results := db.Table("cpod_resource").Delete(r)
	if results.Error != nil {
		glog.Errorf("delete cpod resource error: %s", results.Error.Error())
		return results.Error.Error(), results.Error
	}
	return "", nil
}
