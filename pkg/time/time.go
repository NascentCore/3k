package time

import (
	"time"
)

func GetNearestMinute() time.Time {
	now := time.Now()
	// 将当前时间向下舍入到最近的整分钟
	nearestMinute := time.Date(now.Year(), now.Month(), now.Day(), now.Hour(), now.Minute(), 0, 0, now.Location())
	return nearestMinute
}
