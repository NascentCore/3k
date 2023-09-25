package server

import (
	"io/ioutil"
	"net/http"
	"sxwl/3k/mm/internal/logic"

	"github.com/gin-gonic/gin"
	"github.com/golang/glog"
)

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
