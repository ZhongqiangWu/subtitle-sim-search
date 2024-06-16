# /bin/bash

LOG_DIR='../log/'

if [ ! -d ${LOG_DIR} ]; then
    mkdir ${LOG_DIR}
fi

cd ../src
nohup python3 -u vectorize_subtitle.py > ${LOG_DIR}/vectorization.log 2>&1 &
echo "process started."