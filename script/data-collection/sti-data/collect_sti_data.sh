# $1: start sti id
# $2: end sti id

./profile_sti_exec.py $1 $2
PROFILE_DATA=$(find $SNOWCAT_STORAGE/sti-data/raw/ -name "profile*" -type d -maxdepth 1)
./extract_from_raw_trace.py $PROFILE_DATA
