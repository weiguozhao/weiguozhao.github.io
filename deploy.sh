#!/usr/bin/env bash

echo "start deploy..."
hexo clean && hexo g && hexo d
echo "end deoply..."

message="$1"
if [ -z "${message}" ]; then
    echo "usage: $0 [push message]"
    exit 1
fi

branch=`git symbolic-ref --short -q HEAD`
if [[ $branch -eq "hexo" ]]; then
    echo "start push to hexo branch..."
    git add .
    git commit -m "${message}"
    git push origin hexo
    echo "end push to hexo branch..."
else
    echo "branch must be hexo"
    exit 1
fi

exit 0
