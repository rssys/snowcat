SB_SOURCE_DIR=$MAIN_HOME/tool/snowboard/src
pushd $MAIN_HOME/tool/snowboard/ > /dev/null
# create a folder for installation
SB_INSTALL_DIR=$MAIN_HOME/tool/snowboard/install
if [ -f "$SB_INSTALL_DIR/bin/qemu-system-i386" ]; then
	echo "Snowboard is already installed"
	exit 0
fi
if ! [ -d "$SB_INSTALL_DIR" ]
then
        mkdir $SB_INSTALL_DIR
fi
# compile the source code
cd $SB_SOURCE_DIR > /dev/null
./configure --prefix=$SB_INSTALL_DIR --disable-strip --target-list="i386-softmmu" --disable-pie --disable-smartcard --disable-docs --disable-libiscsi --disable-xen --disable-spice --cc=$GCC_INSTALL/bin/gcc --host-cc=$GCC_INSTALL/bin/gcc --python=python2
make -j`nproc`
make install
popd > /dev/null
