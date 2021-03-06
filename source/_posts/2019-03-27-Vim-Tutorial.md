---
title: Vim Tutorial
mathjax: true
copyright: false
date: 2019-03-27 14:10:09
categories: 小工具
---

**Vim自带的Tutorial:在终端下输入`vimtutor`即可打开vim自带的教程。**

> cite: https://irvingzhang0512.github.io/2018/06/09/vim-tutor/


## 1. 总结
### 1.1. 光标移动

- 普通移动：`h j k l`。
- 到下一个单词开头：`w`，可以添加数字`2w`。
- 到下一个单词结尾：`e`，可以添加数字`2e`。
- 移动到当前行开头：`0`。
- 移动到当前行结尾：`$`。
- 移动光标到文件底部：`G`。
- 移动光标到文件顶部：`gg`。
- 移动到指定的行：`#G`，其中`#`表示数字。

### 1.2. 删除

- 删除一个字符：命令模式，`x`。
- 删除从当前位置，当下一个单词开头的所有字符：`dw`。
- 删除当前行从当前位置开始，到结尾的字符：`d$`。
- 删除当前位置，到当前单词结尾的字符：`de`。
- 可以在dw或de中间添加数字，如`d2w`，`d5e`。
- 删除整行：`dd`，并将数据保存到buffer。
  - 删除多行：在`#dd`前添加数字，`#`表示数字。

### 1.3. 查询与替换

- 查询：在命令模式中输入`/`以及需要查询的内容，回车。
  - 输入`n`查询下一个匹配。
  - 输入`N`查询上一个匹配。
  - 如果要反向查询，则使用`?`而不是`/`。
- 选项：
  - 查询时不区分大小写：`:set ic`, `:set noic`。
  - 查询结果高亮：`:set hls`，`:nohlsearch`。
- 使用`%`查询其对应的字符，如`()`, `{}`, `[]`。
- 替换光标指向的一个字符：`r`（只能替换一个字符）。
  - 在命令模式输入`r`，再输入需要替换的字符，之后就再次进入命令模式。
- 替换多个字符：`R`。
- 替换字符串：
  - `:s/old/new：`只替换一个记录。
  - `:s/old/new/g：`替换当前行所有匹配记录。
  - `:#,#s/old/new/g：`替换指定行中所有匹配的记录。
  - `:%s/old/new/g：`替换文件中所有匹配记录。
  - `:%s/old/new/gc：`替换文件中所有匹配记录，在替换前会询问。

### 1.4. 复制粘贴

- 删除的内容是保存在缓存中，可以通过`p`黏贴。
- 复制：使用`v`，并移动光标，选择需要复制的文本，使用`y`进行复制。
- 粘贴：`p`。
- 获取其他文本中的内容，并复制到本地：`:r TEST`。
  - 也可以复制命令结果，如`:r !ls`。

### 1.5. 添加操作

- 插入字符（进入编辑模式）：`i`。
- 追加字符
  - 在光标指向的位置后添加内容，进入编辑模式：`a`。
  - 在本行末尾添加内容，进入编辑模式：`A`。
- change operator：删除指定位置的数据，并进入编辑模式。
  - `ce`：删除光标到当前单词末尾的数据，并进入编辑模式。
  - `c`与`d`的用法类似，也可以使用`e w $`以及`数字`。

### 1.6. 其他操作

- 退出vim：命令模式，输入`:q!`（退出不保存），输入`:wq`（保存并退出）。
- 查看当前处于文件中的位置：`CTRL-G`。
- 撤销：`u`撤销一个操作，`U`撤销当前行的之前的操作。
  - 撤销的撤销：`CTRL-R`。
- 返回到之前光标的位置：`CTRL-O`，其反操作是`CTRL-I`。

## 2. 分课程总结

### 2.1. 第一课

- 移动光标：命令模式，`h j k l`。
- 退出vim：命令模式，输入`:q!`（退出不保存），输入`:wq`（保存并退出）。
- 删除一个字符：命令模式，`x`。
- 插入字符（进入编辑模式）, `i`。
- 追加字符
  - 在光标指向的位置后添加内容，进入编辑模式：`a`。
  - 在本行末尾添加内容，进入编辑模式：`A`。
- 执行shell命令：`:!ls`。
- 保存到某文件：`:w TEST`。

### 2.2. 第二课

- 删除从当前位置，当下一个单词开头的所有字符：`dw`。
- 删除当前行从当前位置开始，到结尾的字符：`d$`。
- 删除当前位置，到当前单词结尾的字符：`de`。
- 移动光标：
  - 到下一个单词开头：`w`。
  - 到下一个单词结尾：`e`。
  - 移动到当前行开头：`0`。
  - 可以在`w`或`e`前，添加数字。
  - 可以在`dw`或`de`中间添加数字，如`d2w`，`d5e`。
- 删除整行：`dd`，并将数据保存到buffer。
  - 删除多行：在`dd`前添加数字。
- 撤销：`u`撤销一个操作，`U`撤销当前行的之前的操作。
  - 撤销的撤销：`CTRL-R`。

### 2.3. 第三课

- 删除的内容是保存在缓存中，可以通过`p`黏贴。
- 替换光标指向的一个字符：`r`（只能替换一个字符）。
  - 在命令模式输入`r`，再输入需要替换的字符，之后就再次进入命令模式。
- change operator：删除指定位置的数据，并进入编辑模式。
  - `ce`：删除光标到当前单词末尾的数据，并进入编辑模式。
  - `c`与`d`的用法类似，也可以使用`e w $`以及数字。

### 2.4. 第四课

- 查看当前处于文件中的位置：`CTRL-G`。
- 移动光标到文件底部：`G`。
- 移动光标到文件顶部：`gg`。
- 移动到指定的行：`#G`，其中`#`表示数字。
- 查询：在命令模式中输入`/`以及需要查询的内容。
  - 输入`n`查询下一个匹配。
  - 输入`N`查询上一个匹配。
  - 如果要反向查询，则使用`?`而不是`/`。
- 返回到之前光标的位置：`CTRL-O`，其反操作是`CTRL-I`。
- 使用`%`查询其对应的字符，如`()`, `{}`, `[]`。
- 替换字符串：
  - `:s/old/new`：只替换一个记录。
  - `:s/old/new/g`：替换当前行所有匹配记录。
  - `:#,#s/old/new/g`：替换指定行中所有匹配的记录。
  - `:%s/old/new/g`：替换文件中所有匹配记录。
  - `:%s/old/new/gc`：替换文件中所有匹配记录，在替换前会询问。

### 2.5. 第五课

- 执行shell命令：`:!ls`。
- 保存到某文件：`:w TEST`。
- 选择文本：使用`v`，之后移动光标，就可以选择一段文本。
  - 之后若使用`:w TEST`命令，可以将选中的文本保存到指定文件中
- 获取其他文本中的内容，并复制到本地：`:r TEST`。
  - 也可以复制命令结果，如`:r !ls`。

### 2.6. 第六课

- 添加行：`o`光标下添加行，`O`光标上添加行。
- 替换多个字符：`R`。
- 替换一个字符：`r`。
- 复制黏贴：
  - 复制：使用`v`，并移动光标，选择需要复制的文本，使用`y`进行复制。
  - 粘贴：`p`。
- 选项：
  - 查询时不区分大小写：`:set ic`, `:set noic`。
  - 查询结果高亮：`:set hls`，`:nohlsearch`

