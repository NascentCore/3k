// NO_TEST_NEEDED
package router

import (
	"sxwl/mm/internal/server"

	"github.com/gin-gonic/gin"
)

func CpodRouter(c *gin.Engine) {
	v1 := c.Group("/api/v1/")
	{
		v1.POST("cpod/job", server.CpodJob)
		v1.POST("cpod/job/result", server.CpodJobResult)
		v1.POST("cpod/resource", server.CpodResource)
	}
}
