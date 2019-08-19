#!/usr/bin/env bash

echo "step 1: clone source files"
echo "git clone https://github.com/weiguozhao/weiguozhao.github.io.git\n"

echo "step 2: install node.js"
echo "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.34.0/install.sh | sh"
echo "nvm install stable\n"

echo "step 3: install hexo-cli"
echo "npm install -g hexo-cli\n"

echo "step 4: install dependencies"
echo "cd weiguozhao.github.io && npm install\n"


