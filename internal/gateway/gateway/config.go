package gateway

type Router struct {
	Path   string
	ToPath string `json:",optional"`
	Auth   bool   `json:",optional"`
	Server string
}

type Config struct {
	Servers map[string]string
	Routers []Router
}
