# default values if not provided via command line arguments
MODEL=${1:-ann}
STRATEGY=${2:-k-fold}
RUNS=${3:-100}
MIN_R2=${4:-0.9}

echo " Model: $MODEL"
echo " Strategy: $STRATEGY"
echo " Total Runs: $RUNS"
echo " Min R²: $MIN_R2"

for i in $(seq 1 $RUNS); do
    echo -e "\nRun $i/$RUNS Executing model $MODEL"
    
    python main.py -m "$MODEL" -s "$STRATEGY" -o cost co2 --min-r2 "$MIN_R2"
    
    # check if the program exited with an error
    if [ $? -ne 0 ]; then
        echo "Error encountered on run $i."
        exit 1
    fi
done