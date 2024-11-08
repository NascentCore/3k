package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"time"

	"github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	podresourcesapi "k8s.io/kubelet/pkg/apis/podresources/v1alpha1"
)

var (
	connectionTimeout = 10 * time.Second
	podName           string
)

func main() {
	flag.StringVar(&podName, "podName", "exporter", "the pod name")
	flag.Parse()

	socketPath := "/var/lib/kubelet/pod-resources/kubelet.sock"
	_, err := os.Stat(socketPath)
	if os.IsNotExist(err) {
		logrus.Info("No Kubelet socket, ignoring")
		return
	}

	c, cleanup, err := connectToServer(socketPath)
	if err != nil {
		logrus.Info("Cannot connect to kubelet")
		return
	}
	defer cleanup()

	client := podresourcesapi.NewPodResourcesListerClient(c)

	ctx, cancle := context.WithTimeout(context.Background(), connectionTimeout)
	defer cancle()

	resp, err := client.List(ctx, &podresourcesapi.ListPodResourcesRequest{})
	if err != nil {
		logrus.Error(err)
	}

	deviceToPodMap := make(map[string]PodInfo)
	for _, pod := range resp.GetPodResources() {
		for _, container := range pod.Containers {
			for _, devices := range container.Devices {
				for _, deviceID := range devices.DeviceIds {
					deviceToPodMap[deviceID] = PodInfo{
						Name:         pod.Name,
						Namespace:    pod.Namespace,
						Container:    container.Name,
						ResourceName: devices.ResourceName,
					}
				}
			}
		}
	}

	logrus.Info(deviceToPodMap)

	logrus.Info("===Print current pod bind devices===")
	for d, pod := range deviceToPodMap {
		if pod.Name == podName {
			logrus.Infof("DeviceID: %s, Pod: %s, Container: %s, ResourceName: %s", d, pod.Name, pod.Container, pod.ResourceName)
		}
	}

	// Wait for a signal to terminate the program
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, os.Interrupt)
	<-quit
}

type PodInfo struct {
	Name         string
	Namespace    string
	Container    string
	ResourceName string
}

func connectToServer(socket string) (*grpc.ClientConn, func(), error) {
	ctx, cancel := context.WithTimeout(context.Background(), connectionTimeout)
	defer cancel()

	conn, err := grpc.DialContext(ctx,
		socket,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
		grpc.WithContextDialer(func(ctx context.Context, addr string) (net.Conn, error) {
			d := net.Dialer{}
			return d.DialContext(ctx, "unix", addr)
		}),
	)
	if err != nil {
		return nil, func() {}, fmt.Errorf("failure connecting to '%s'; err: %w", socket, err)
	}

	return conn, func() { conn.Close() }, nil
}
