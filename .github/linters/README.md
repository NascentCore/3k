# Linters

This is the default location for storing super-linter's config files for
[individual linters](https://github.com/github/super-linter#template-rules-files).

The names of these config files should not be changed.

All of them are dot-files that are hidden by default. This also follows the
official guideline.

.shellcheckrc does not work for super-linter,
see [super-linter/issues/4645](https://github.com/super-linter/super-linter/issues/4645)
to ignore certain error, place a .shellcheckrc file under the target directory
to ignore that error for files under that directory (non-recursively).
