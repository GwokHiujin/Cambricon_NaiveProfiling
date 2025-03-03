# set -x
set -e

CUR_DIR=$(
  cd $(dirname $0)
  pwd
)

usage() {
    echo "Usage: "
    echo "-------------------------------------------------------------"
    echo "|  $0 [0-1]"
    echo "|  parameter1: 0)performance, 1)from scratch,"
    echo "|  eg. ./launch.sh 0"
    echo "|      which means launch performace db ."
    echo "-------------------------------------------------------------"
}

# Parameters Validation
if [[ $1 =~ ^[0-1] ]]; then
    echo "Parameters Exact."
else
    echo "[ERROR] Unknown Parameter."
    usage
    exit 1
fi


OUTPUT_DIR=$CUR_DIR/output
LOG_DIR=$OUTPUT_DIR/log
REPORT_DIR=$OUTPUT_DIR/report
SUMMARY_DIR=$OUTPUT_DIR/summary
_HARD_FILE=$OUTPUT_DIR/hard_info.yaml
_SOFT_FILE=$OUTPUT_DIR/soft_info.yaml

SUPERSET_HOME=$CUR_DIR

# TODO(guwei) Modify after version release
PT_SOFT_NAME=""
PT_HARD_NAME=""
PT_CODE_REPO=""
PT_CODE_COMMIT=""
PT_CODE_LINK=""
DEV_NAME=""
DB_YAML=""
INPUT_LOG=""

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}
mkdir -p ${REPORT_DIR}
mkdir -p ${SUMMARY_DIR}

upload_db() {
  for file in $(ls ${REPORT_DIR}/*yaml); do
    cndb_submit \
      --db_file $CUR_DIR/config/${DB_YAML} \
      --load_file ${file} \
      --log-level INFO \
      --db_store_type update
  done
}

prepare() {

  PT_CODE_REPO=$(
    cd $SUPERSET_HOME
    git remote get-url --push origin
  )
  PT_CODE_LINK="${PT_CODE_REPO}"
}

usage() {
  echo "Usage:"
  echo "    ./$(basename $0) -s <software name> -d <hardware name> -b <pytorch_model home>"
  exit -1
}

while getopts 's:d:b:' OPT; do
  case $OPT in
  s) PT_SOFT_NAME="$OPTARG" ;;
  d) PT_HARD_NAME="$OPTARG" ;;
  b) SUPERSET_HOME="$OPTARG" ;;
  *) usage ;;
  esac
done

echo "Step 1: prepare for cndb"

if [ $1 -eq 0 ]; then
  DB_YAML="perf_db.yaml"
  INPUT_LOG="input"
else
  DB_YAML="from_scratch_db.yaml"
  INPUT_LOG="metrics_log"
fi
prepare


echo "Step 2: convert benchmark_log to cndb format"
python dump.py \
  -i ${SUPERSET_HOME}/${INPUT_LOG} \
  -o ${REPORT_DIR} \
  --code_link ${PT_CODE_LINK}


echo "Step 3: upload cndb"
upload_db

echo "Step 4: deleta output files"
rm ${OUTPUT_DIR} -fr

