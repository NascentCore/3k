# 增加数据盘

1. 登录gcp控制台，给 vm instance 增加一块数据盘

2. 登录 instance 查看数据盘
```
$ lsblk
NAME    MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
sda       8:0    0   20G  0 disk
├─sda1    8:1    0 19.9G  0 part /
├─sda14   8:14   0    3M  0 part
└─sda15   8:15   0  124M  0 part /boot/efi
sdb       8:16   0  100G  0 disk /data
```

3. 格式化数据盘文件系统
```
sudo mkfs.ext4 /dev/sdb
```

4. 编辑 /etc/fstab 文件，增加开机挂载数据盘
```
/dev/sdb /data ext4 rw 0 0
```

5. 挂载数据盘
```
sudo mount -a
```

6. 当前占用空间的数据都在 /root、/home/deploy
   目录下，将数据移动到数据盘并创建软链接
```
sudo mv /home/deploy /data
sudo mv /root /data
sudo ln -s /data/deploy /home/deploy
sudo ln -s /data/root /root
```
