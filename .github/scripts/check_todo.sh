#!/bin/bash

echo "Checking TODO style"

if grep 'todo' home cli manager -R; then
  echo "TODO should all be upper case, not todo"
  exit 1
fi

if grep 'todo:' home cli manager -R; then
  echo "TODO should have assignee, TODO(user):, not TODO:"
  exit 1
fi
