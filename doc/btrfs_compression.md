# How to enable btrfs compression on a disk?

The btrfs compression can significantly reduce the actual disk usage.

## Prerequisite

- An empty disk

## Instructions

1. Create the btrfs filesystem on the disk (assuming the target disk is `/dev/sdd`)

   ```bash
   sudo mkfs.btrfs /dev/sdd
   ```

2. Mount the disk

   ```bash
   #mkdir $dst_dirpath
   mkdir snowcat-storage/
   mount -o compress=zstd:4 /dev/sdd snowcat-storage/
   ```

   Note: `zstd:4` means we use level=4 compression. There are in total 10 levels to choose, through which you can trade-off between the speed and disk usage.

3. Use the disk!