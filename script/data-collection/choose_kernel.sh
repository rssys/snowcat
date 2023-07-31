KERNEL_DIRNAME=$1
KERNEL_DIRPATH=$MAIN_HOME/data/kernel/$KERNEL_DIRNAME
if [ ! -d "$KERNEL_DIRPATH" ]; then
        echo "data for $KERNEL_DIRNAME is not available"
        echo "only supports linux-kernel-6.1 for now"
        exit 1
fi

pushd $KERNEL_DIRPATH > /dev/null
./download.sh
RET=$?
if [ $RET -ne 0 ]; then
        echo "[Error] kernel data is not downloaded correctly. Try again?"
        exit 1
fi
export BZIMAGE_FILEPATH=$KERNEL_DIRPATH/bzImage
export SNAPSHOT_FILEPATH=$KERNEL_DIRPATH/snapshot.img
export KERNEL_INFO_DIR=$KERNEL_DIRPATH
popd > /dev/null
