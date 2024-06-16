# /bin/bash

LOG_DIR='../log/'

if [ ! -d ${LOG_DIR} ]; then
    mkdir ${LOG_DIR}
fi

QUERY="人生无悔"

cd ../src
nohup python3 -u seach_subtitle.py --query=${QUERY} > ${LOG_DIR}/search.log 2>&1 &
echo "process started."
