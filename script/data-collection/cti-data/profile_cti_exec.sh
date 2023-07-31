#!/bin/bash

error_msg(){
        echo [PROFILE_CTI_EXEC.SH] ERROR: $1
        echo [PROFILE_CTI_EXEC.SH] ERROR: exiting!
        exit 1
}

log_msg(){
        echo [PROFILE_CTI_EXEC.SH] $1
}
## Default directories values: update or override them ##
#SKI_TMP=${SKI_TMP-"/dev/shm/ski-user/tmp/"}
SKI_TMP=${SKI_TMP-"/dev/shm/snowboard/"}
if ! [ -d "$SKI_OUTPUT_DIR" ]
then
        echo "directory not exists, creating $SKI_OUTPUT_DIR"
        mkdir $SKI_OUTPUT_DIR
fi

SKI_OUTPUT_DIR="$SKI_OUTPUT_DIR/collect_data-`date +'%Y-%m-%d-%H-%M-%S'`"
if ! [ -d "$SKI_OUTPUT_DIR" ]
then
        echo "dir not exists, creating $SKI_OUTPUT_DIR"
        mkdir $SKI_OUTPUT_DIR
fi
export SKI_OUTPUT_DIR

log_file="$SKI_OUTPUT_DIR/run-ski-`date +'%Y-%m-%d-%H-%M-%S'`.txt"
echo $log_file
exec &> >(tee "$log_file")
echo "This will be logged to the file and to the screen"

SKI_BINARY=$SKI_INSTALL_DIR/bin/qemu-system-i386
SKI_VM_FILENAME=$SNAPSHOT_FILEPATH
VMM_RAM_MB=2048
VMM_CPUS=4
VMM_BOOTLOADER_LINUX_APPEND="root=/dev/sda1 rw -verbose console=tty0 console=ttyS0"
VMM_MISC_OPTIONS="-rtc base=utc,clock=vm -qmp tcp:localhost:10000,server,nowait -net nic -net user,hostfwd=tcp::10001-:22"
VMM_SKI_ARGS="-ski 0,input_number=210:210,preemptions=119:97"

# some verification
if ! [ -x $SKI_BINARY ] ; then error_msg "Unable to find the binary at $SKI_BINARY."; fi
if [ -z "$SKI_VM_FILENAME" ] || ! [ -f "$SKI_VM_FILENAME" ] ; then  error_msg "Need to set SKI_VM_FILENAME to a valid snapshot"; fi
if ! [ -d "$SKI_OUTPUT_DIR" ] ; then  mkdir $SKI_OUTPUT_DIR || error_msg "Need create the output directory (SKI_OUTPUT_DIR=$SKI_OUTPUT_DIR)."; fi

# Enable core dumps
ulimit -c unlimited
ulimit -a
# Note that coredumps can be extremely large, specially if not filtered because of the large address space
# Running concurrent test by resuming from a snapshot

VMM_SNAPSHOT=ski-vm-XXX
VMM_HDA_FILENAME=$SKI_TMP/tmp.$$.img
VMM_SERIAL_FILENAME=file:$SKI_OUTPUT_DIR/console.txt

rm -r $SKI_TMP/tmp*img
mkdir -p $SKI_TMP
log_msg "Copying the VM image to tmp"
cp $SKI_VM_FILENAME $VMM_HDA_FILENAME || error_msg "Unable to copy the VM image to the temporary directory (SKI_TMP=$SKI_TMP)!"


# Parameters that are expected to be provided by the user to SKI (e.g., when calling ./run-ski.sh)
export SKI_PREEMPTION_BY_EIP=1
export SKI_PREEMPTION_BY_ACCESS=0
export SKI_TEST_CPU_1_MODE=3
export SKI_TEST_CPU_2_MODE=3
export SKI_RACE_DETECTOR_ENABLED=1
export SKI_TRACE_INSTRUCTIONS_ENABLED=1
export SKI_TRACE_MEMORY_ACCESSES_ENABLED=1
export SKI_SIMPLIFIED_TRACING=1
export SKI_TRACE_WRITE_SET_ENABLED=0
export SKI_TRACE_READ_SET_ENABLED=0
export SKI_TRACE_EXEC_SET_ENABLED=0
export SKI_TRACE_SET_CPU=0
# Other SKI parameters
export SKI_RESCHEDULE_POINTS=1
export SKI_RESCHEDULE_K=1
export SKI_FORKALL_ENABLED=1
export SKI_WATCHDOG_SECONDS=120
export SKI_QUIT_HYPERCALL_THRESHOLD=1
export SKI_OUTPUT_DIR_PER_INPUT_ENABLED=1
export SKI_DEBUG_START_SLEEP_ENABLED=0
export SKI_DEBUG_CHILD_START_SLEEP_SECONDS=0
export SKI_DEBUG_CHILD_WAIT_START_SECONDS=0
export SKI_DEBUG_PARENT_EXECUTES_ENABLED=0
export SKI_DEBUG_EXIT_AFTER_HYPERCALL_ENABLED=0
export SKI_MEMFS_ENABLED=1
export SKI_MEMFS_TEST_MODE_ENABLED=0
export SKI_MEMFS_LOG_LEVEL=1
export SKI_PRIORITIES_FILENAME=$SKI_INSTALL_DIR/../testcase.priorities
export SKI_KERNEL_FILENAME=$BZIMAGE_FILEPATH
export SKI_CORPUS_DIR=$MAIN_HOME/data/sti/data

log_msg "Running command"
$SKI_BINARY -m $VMM_RAM_MB -smp $VMM_CPUS -loadvm $VMM_SNAPSHOT -kernel $SKI_KERNEL_FILENAME -hda $VMM_HDA_FILENAME -serial $VMM_SERIAL_FILENAME $VMM_MISC_OPTIONS $VMM_SKI_ARGS
RET=$?

log_msg "Removing the VM image in tmp.."
rm $VMM_HDA_FILENAME

exit $RET
