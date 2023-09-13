#!/usr/bin/env bash -xe

# Based on
# https://rook.io/docs/rook/v1.12/Getting-Started/ceph-teardown/#zapping-devices

if [[ $# < 1 ]]; then
  echo "Need DISK name $0 <DISK-name>, exit ..."
  exit 1
fi

DISK="$1"

# Zap the disk to a fresh, usable state (zap-all is important, b/c MBR has to be clean)
sgdisk --zap-all $DISK

# Wipe a large portion of the beginning of the disk to remove more LVM metadata that may be present
dd if=/dev/zero of="$DISK" bs=1M count=100 oflag=direct,dsync

# SSDs may be better cleaned with blkdiscard instead of dd
blkdiscard $DISK

# Inform the OS of partition table changes
partprobe $DISK

# This command hangs on some systems: with caution, 'dmsetup remove_all --force' can be used
ls /dev/mapper/ceph-* | xargs -I% -- dmsetup remove %

# ceph-volume setup can leave ceph-<UUID> directories in /dev and /dev/mapper (unnecessary clutter)
rm -rf /dev/ceph-*
rm -rf /dev/mapper/ceph--*
