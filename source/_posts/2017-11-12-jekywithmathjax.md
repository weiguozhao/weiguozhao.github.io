---
layout: post
title: Jekyll With Mathjax
date: 2017-11-12 10:10 +0800
categories: 工具
tags:
- 工具
mathjax: true
copyright: true
---

<!-- 如果该blog有其他图片代码文件，需在/posts_res/2018-01-01-template/存放 -->

## <center>设置Jekyll和Github上的Mathjax</center>

如何做呢？

很简单。

* 1.到你的Jekyll目录下，默认路径为`username.github.io/_includes/head.html`

* 2.将下列代码复制粘贴到里面（**注意要放到<head\> ... </head\>之间**）

```js
<head>
    ...
    <script type="text/x-mathjax-config"> MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); </script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
                inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                processEscapes: true
            }
        });
        </script>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
</head>
```


* 3.完成！

* 4.测试一下吧！

$$ E = m\cdot c^2 \label{eq:mc2}$$


Thanks
> [csega](http://csega.github.io/mypost/2017/03/28/how-to-set-up-mathjax-on-jekyll-and-github-properly.html)