// NO_TEST_NEEDED
package server

import (
	"io/ioutil"
	"net/http"
	"sxwl/mm/internal/logic"

	"github.com/gin-gonic/gin"
	"github.com/golang/glog"
)

// @Summary: cpod jobs scheduler function
// @Accept json
// @Produce json
// @Success 200 {object} db.JobScheduler
// @Failure 500
// @Router /api/v1/cpod/job [post]
func CpodJob(c *gin.Context) {
	body, err := ioutil.ReadAll(c.Request.Body)
	glog.V(5).Infof("the cpod request body is %s", string(body))
	if err != nil {
		glog.Errorf("Error reading cpod job body: %v", err)
		c.JSON(http.StatusInternalServerError, "Error reading cpod job body")
		return
	}
	result, err := logic.CpodJobLogical(body)
	if err != nil {
		glog.Errorf("Error executing cpod job: %v", err)
		c.JSON(http.StatusInternalServerError, err.Error())
		return
	}
	c.JSON(http.StatusOK, result)
	return
}

// @Summary: cpod jobs scheduler result function
// @Accept json
// @Produce json
// @Param data body req true db.JobScheduler
// @Success 200
// @Failure 500
// @Router /api/v1/cpod/job/result [post]
func CpodJobResult(c *gin.Context) {
	body, err := ioutil.ReadAll(c.Request.Body)
	glog.V(5).Infof("the cpod result request body is %s", string(body))
	if err != nil {
		glog.Errorf("Error reading cpod job reuslt body: %v", err)
		c.JSON(http.StatusInternalServerError, "Error reading cpod job result body")
		return
	}
	result, err := logic.CpodJobLogical(body)
	if err != nil {
		glog.Errorf("Error executing cpod job result: %v", err)
		c.JSON(http.StatusInternalServerError, err.Error())
		return
	}
	c.JSON(http.StatusOK, result)
}

// @Summary: cpod resource function
// @Accept json
// @Produce json
// @Param data body req true db.CpodResources
// @Success 200
// @Failure 500
// @Router /api/v1/cpod/resource [post]
func CpodResource(c *gin.Context) {
	body, err := ioutil.ReadAll(c.Request.Body)
	glog.V(5).Infof("the cpod resource request body is %s", string(body))
	if err != nil {
		glog.Errorf("Error reading cpod Resource body: %v", err)
		c.JSON(http.StatusInternalServerError, "Error reading cpod Resource body")
		return
	}
	result, err := logic.CpodResourceLogical(body)
	if err != nil {
		glog.Errorf("Error executing cpod Resource: %v", err)
		c.JSON(http.StatusInternalServerError, err.Error())
		return
	}
	c.JSON(http.StatusOK, result)
	return
}
