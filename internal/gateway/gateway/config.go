package gateway

type Router struct {
	Path   string
	ToPath string `yaml:",omitempty"`
	Auth   bool   `yaml:",omitempty"`
	Server string
}

type Config struct {
	Servers map[string]string
	Routers []Router
}
