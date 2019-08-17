---
title: hexo-NexT的Local Search转圈问题解决记录
mathjax: true
copyright: true
date: 2019-03-28 20:20:45
categories: 小工具
---

> cite: https://guahsu.io/2017/12/Hexo-Next-LocalSearch-cant-work/

有时候更新完文章之后，莫名其妙地Local Search不能用了，这是大部分是因为文章中有非法字符`bs`

## Step1. 检查问题来源

由于使用的是localSearch，在使用`hexo g`的时候，会在public里面生成search.xml作为搜索主体，
之后使用一些在线验证XML的网站取检查，把search.xml的内容全部拿过去检查，[这里可以用来检查](https://www.xmlvalidation.com/)。

一般都会出来一个这个问题：

![question](/posts_res/2019-03-28-hexo-NexT的Local-Search转圈问题解决记录/1.png)

总之，问题就是因为多了这个backspace字符！


## Step2. 显示看不到的backspace字符

我是使用VSCODE，开启方式是到在设定中选中renderControlCharacters

![answer](/posts_res/2019-03-28-hexo-NexT的Local-Search转圈问题解决记录/2.png)

打开之后你就可以看到那个backspace字符了……

![location](/posts_res/2019-03-28-hexo-NexT的Local-Search转圈问题解决记录/3.png)

## Step3. 搜索并替换

backspace的unicode是\u0008，

而VSCODE的搜索正则表达式使用的是Rust要輸入\x{0008}才可以，

其实直接把那个很小的bs框起來复制放到搜索框中即可！！！

## Step4. 重新生成部署

重新在小站目录下执行`hexo d -g`，然后看看线上的Local Search吧！
