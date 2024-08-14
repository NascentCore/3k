package gateway

type Router struct {
	Path   string
	ToPath string `json:",optional"` //nolint:staticcheck
	Auth   bool   `json:",optional"` //nolint:staticcheck
	Server string
}

type Config struct {
	Servers map[string]string
	Routers []Router
}
