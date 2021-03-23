---
title: MacOS支持写NTFS格式硬盘
tags:
  - MacOs
  - NTFS
  - Shell
mathjax: true
comments: false
copyright: true
date: 2021-03-23 16:49:25
categories: 小工具
---


MacOS 从雪豹开始已经支持写 NTFS 格式硬盘，但是系统默认挂载是只读的，可通过手动 `mount` 打开写入支持：

接入移动硬盘，待自动挂载完成后，在 terminal 中输入命令 `mount`，可看到类似如下结果：

```shell
mount
/dev/disk1s1 on / (apfs, local, journaled)
devfs on /dev (devfs, local, nobrowse)
/dev/disk1s4 on /private/var/vm (apfs, local, noexec, journaled, noatime, nobrowse)
map -hosts on /net (autofs, nosuid, automounted, nobrowse)
map auto_home on /home (autofs, automounted, nobrowse)
/dev/disk2s1 on /Volumes/Elements (ntfs, local, nodev, nosuid, read-only, noowners)
```

注意其中这行：

```shell
/dev/disk2s1 on /Volumes/Elements (ntfs, local, nodev, nosuid, read-only, noowners)
```

其中 `/dev/disk2s1` 就是移动硬盘（后面需要用到这个路径），可看到其后有 `ntfs` 字样，如果不确定，可以将移动硬盘弹出，然后再输入 `mount`，对比输出结果即可确定哪个设备是移动硬盘。

卸载自动挂载的移动硬盘：

```shell
sudo umount /dev/disk2s1
```

建立挂载点，这里我选择挂载到桌面上：

```shell
mkdir -p ~/Desktop/Elements
```

以读写方式重新挂载：

```shell
sudo mount -t ntfs -o rw,auto,nobrowse /dev/disk2s1 ~/Desktop/Elements
```

现在，移动硬盘就以可读写的形式出现在桌面上了。如果要弹出硬盘可以右键它选择 `弹出` 即可。
