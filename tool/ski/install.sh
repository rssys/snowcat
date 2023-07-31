SKI_SOURCE_DIR=$MAIN_HOME/tool/ski/src
pushd $MAIN_HOME/tool/ski/ > /dev/null
# create a folder for installation
SKI_INSTALL_DIR=$MAIN_HOME/tool/ski/install
if [ -f "$SKI_INSTALL_DIR/bin/qemu-system-i386" ]; then
	echo "snowboard is already installed"
	exit 0
fi
if ! [ -d "$SKI_INSTALL_DIR" ]
then
        mkdir $SKI_INSTALL_DIR
fi
# compile the source code
cd $SKI_SOURCE_DIR > /dev/null
./configure --prefix=$SKI_INSTALL_DIR --disable-strip --target-list="i386-softmmu" --disable-pie --disable-smartcard --disable-docs --disable-libiscsi --disable-xen --disable-spice --cc=$GCC_INSTALL/bin/gcc --host-cc=$GCC_INSTALL/bin/gcc --python=python2
make -j`nproc`
make install
popd > /dev/null
