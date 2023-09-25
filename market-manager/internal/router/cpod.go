package router

import (
	"sxwl/3k/mm/internal/server"

	"github.com/gin-gonic/gin"
)

func CpodRouter(c *gin.Engine) {
	v1 := c.Group("/api/v1/cpod")
	{
		v1.POST("/job", server.CpodJob)
		v1.POST("/job/result", server.CpodJobResult)
		v1.POST("/resource", server.CpodResource)
	}
}
