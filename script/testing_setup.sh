log_msg(){
	echo "[setup.sh] LOG: $1"
}

ulimit -n 20480
export SNOWCAT_STORAGE=$1

sudo apt-get install build-essential python gcc python3-pip
sudo apt install libglib2.0-dev libfdt-dev libpixman-1-dev zlib1g-dev libgtk-3-dev
sudo apt-get install gcc-multilib g++-multilib

DIR=$(dirname $BASH_SOURCE)
echo $DIR
export MAIN_HOME=$(realpath $DIR/../)

# Compile gcc-5.4
$MAIN_HOME/tool/gcc/install.sh
RES=$?
if [ $RES -ne 0 ]; then 
	log_msg "GCC compilation failed ($RES)." ;  
fi
source $MAIN_HOME/tool/gcc/env.sh

# Compile the qemu emulator SKI and SNOWBOARD
$MAIN_HOME/tool/ski/install.sh
RES=$?
if [ $RES -ne 0 ]; then 
	log_msg "SKI compilation failed ($RES)." ;  
fi
source $MAIN_HOME/tool/ski/env.sh

$MAIN_HOME/tool/snowboard/install.sh
RES=$?
if [ $RES -ne 0 ]; then
	log_msg "Snowboard compilation failed ($RES)." ;
fi
source $MAIN_HOME/tool/snowboard/env.sh

source $MAIN_HOME/script/data-collection/choose_kernel.sh linux-kernel-6.1
