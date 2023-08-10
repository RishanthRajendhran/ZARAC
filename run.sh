#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH -o outputs-%j

DEEPSPEEDTRAIN=false
DEEPSPEEDTEST=false
DIRECT=false
PEFT=false

LOGFILE="stdout"
EPISODES=5
STARTEPISODE=0
ENDEPISODE=-1
DATAPPREFIX="/uufs/chpc.utah.edu/common/home/u1419542/ZARA/datasets/ecqa/episodes/"
TRAINNAME="train.jsonl"
TESTNAME="test.jsonl"
MODEL="unifiedqa"
MODELSIZE="large"
DATASET="ecqa"
TRAINPROMPT="None"
TESTPROMPT="None"
NUMEPOCHS=1
BATCHSIZE=4
LEARNINGRATE=3e-4
SAVEPATH="/uufs/chpc.utah.edu/common/home/u1419542/scratch/ZARA/model/"
MAXSTEPS=300
MAXSHOTS=9
LEARNINGRATESCHEDULER=-1


while getopts 'a:b:c:de:f:g:h:jl:m:n:o:p:q:s:t:uv:w:x:y:z:' opt; do
  case "$opt" in
    a)   LOGFILE="$OPTARG"  ;;
    b)   BATCHSIZE="$OPTARG"  ;;
    c)   LEARNINGRATE="$OPTARG"  ;;
    d)   DEEPSPEEDTEST=true   ;;
    e)   NUMEPOCHS="$OPTARG"  ;;
    f)   MAXSTEPS="$OPTARG"  ;;
    g)  DATASET="$OPTARG"   ;;
    h)  DATAPPREFIX="$OPTARG"   ;;
    j)   DEEPSPEEDTRAIN=true   ;;
    l)  MODELSIZE="$OPTARG"   ;;
    m)   MODEL="$OPTARG"     ;;
    n)   DIRECT=true     ;;
    o)   MAXSHOTS="$OPTARG"  ;;
    p)   TRAINPROMPT="$OPTARG"  ;;
    q)   TESTPROMPT="$OPTARG"  ;;
    s)  LEARNINGRATESCHEDULER="$OPTARG"  ;;
    t) TRAINNAME="$OPTARG" ;;
    u)   PEFT=true     ;;
    v) TESTNAME="$OPTARG" ;;
    w)  SAVEPATH="$OPTARG" ;;
    x) EPISODES="$OPTARG" ;;
    y)  STARTEPISODE="$OPTARG" ;;
    z)  ENDEPISODE="$OPTARG" ;;
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done

export PYTHONPATH=/scratch/general/vast/u1419542/miniconda3/envs/starcEnv/bin/python
source /scratch/general/vast/u1419542/miniconda3/etc/profile.d/conda.sh
conda activate starcEnv

wandb disabled
# mkdir /scratch/general/vast/u1419542/huggingface_cache
export HF_HOME=/scratch/general/vast/u1419542/huggingface_cache
export HF_DATASETS_CACHE="/scratch/general/vast/u1419542/huggingface_cache"
export TRANSFORMERS_CACHE="/scratch/general/vast/u1419542/huggingface_cache"

ADDITIONAL=""
LAUNCHER="python3"
if [ "$DEEPSPEEDTRAIN" = true ] ; then
    ADDITIONAL="${ADDITIONAL} -deepSpeedTrain"
    LAUNCHER="deepspeed"
fi ;

if [ "$DEEPSPEEDTEST" = true ] ; then
    ADDITIONAL="${ADDITIONAL} -deepSpeedTest"
    LAUNCHER="deepspeed"
fi ;
if [ "$DIRECT" = true ] ; then
    ADDITIONAL="${ADDITIONAL} -direct"
fi ;

if [ "$PEFT" = true ] ; then
    ADDITIONAL="${ADDITIONAL} -peft"
fi ;

$LAUNCHER trainTest.py -episodes $EPISODES -startEpisode $STARTEPISODE -endEpisode $ENDEPISODE -dataPrefix $DATAPPREFIX -trainName $TRAINNAME -testName $TESTNAME -model $MODEL -size $MODELSIZE -dataset $DATASET -maxShots $MAXSHOTS -trainPrompt $TRAINPROMPT -testPrompt $TESTPROMPT -numEpochs $NUMEPOCHS -maxSteps $MAXSTEPS -batchSize $BATCHSIZE -learningRate $LEARNINGRATE -savePath $SAVEPATH -log $LOGFILE -lrScheduler $LEARNINGRATESCHEDULER $ADDITIONAL