package logic

import (
	"errors"
	"github.com/golang/glog"
	"sort"
	"sxwl/mm/internal/common"
)

func SelectCpods(cpods []common.CpodResourceInfo) (string, error) {
	if len(cpods) == 0 {
		return "", errors.New("cpods is empty")
	}
	// TODO: select the best cpod
	sort.Slice(cpods, func(i, j int) bool {
		if cpods[i].GpuFree != cpods[j].GpuFree {
			return cpods[i].GpuFree < cpods[j].GpuFree
		}
		return cpods[i].GpuFree < cpods[j].GpuFree
	})
	glog.V(5).Infof("sort cpods: %+v", cpods)
	return "", nil
}
