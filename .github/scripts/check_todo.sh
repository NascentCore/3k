#!/bin/bash

echo "Checking TODO style"

if grep 'todo' cli manager tools -R; then
  echo "TODO should all be upper case, not todo"
  exit 1
fi

if grep 'todo:' cli manager tools -R; then
  echo "TODO should have assignee, TODO(user):, not TODO:"
  exit 1
fi
