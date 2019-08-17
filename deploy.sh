#!/usr/bin/bash

if [[ $1 -eq '' ]]; then
    echo "usage: sh deploy.sh [push message]"
    exit 1
fi

echo "start deploy..."
hexo clean && hexo g && hexo d
echo "end deoply..."

branch=`git symbolic-ref --short -q HEAD`
if [[ $branch -eq "hexo" ]]; then
    echo "start push to hexo branch..."
    git add .
    git commit -m "$1"
    git push
    echo "end push to hexo branch..."
else
    echo "branch must be hexo"
    exit 1
fi
exit 0
