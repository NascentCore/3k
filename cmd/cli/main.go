package main

// NO_TEST_NEEDED

import (
	"log"
	"os"
	"sxwl/3k/pkg/utils/errors"

	"github.com/urfave/cli/v2"
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
