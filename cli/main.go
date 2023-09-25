package main

import (
	"log"
	"os"

	"github.com/urfave/cli/v2"

	"sxwl/3k/common/errors"
)

// This is a CLI that does not do much.
// Used as the basis for further development.
func main() {
	app := &cli.App{
		Name:  "3k",
		Usage: "log in onto the NascentCore.AI platform",
		Action: func(*cli.Context) error {
			return errors.UnImpl("login")
		},
	}

	if err := app.Run(os.Args); err != nil {
		log.Fatal(err)
	}
}
