package main

import (
	"log"
	"os"

	"github.com/urfave/cli/v2"

	"sxwl/3k/common/errors"
)

func main() {
	app := &cli.App{
		Name:  "login",
		Usage: "log in onto the NascentCore.AI platform",
		Action: func(*cli.Context) error {
			return errors.UnImpl("login")
		},
	}

	if err := app.Run(os.Args); err != nil {
		log.Fatal(err)
	}
}
