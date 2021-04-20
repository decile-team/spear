#!/usr/bin/env bash
logdir=logs/my_TREC
mkdir -p $logdir # logs are dumped here

# Keep any one group of flags (4 consecutive lines) active at any time and run the corresponding experiment

# USE THIS FOR IMPLY LOSS
declare -a arr=("implication") # ImplyLoss (Our method) in Table 2 Column2 (Question) (https://openreview.net/pdf?id=SkeuexBtDr)
declare -a gamma_arr=(0.1)
declare -a lamda_arr=(0.1) # not actually used
declare -a model_id=(1 2 3 4 5) # (5 independent models were trained and numbers obtained were averaged)

# USE THIS FOR POSTERIOR REG.
# declare -a arr=("pr_loss") # Posterior Reg. in Table2 Column2 (Question) 
# declare -a gamma_arr=(0.001)
# declare -a lamda_arr=(0.1) # not actually used
# declare -a model_id=(1 2 3 4 5 6 7 8 9 10) # (5 independent models were trained and numbers obtained were averaged)

# USE THIS FOR L+Usnorkel
# declare -a arr=("label_snorkel") # L+Usnorkel in Table2 Column2 (Question)
# declare -a gamma_arr=(0.01)
# declare -a lamda_arr=(0.1) # not actually used
# declare -a model_id=(1 2 3 4 5 6 7 8 9 10)
   
# USE THIS FOR L+Umaj and Nosie-Tolerant
# declare -a arr=("gcross") 
# declare -a gamma_arr=(0.001)
# declare -a lamda_arr=(0 0.9) # 0 for L+Umaj and 0.9 for Noise-tolerant in Table 2 Column2 (Question)
# declare -a model_id=(1 2 3 4 5 6 7 8 9 10)

# USE THIS FOR Snorkel-Noise-Tolerant
# declare -a arr=("gcross_snorkel")
# declare -a gamma_arr=(0.1)
# declare -a lamda_arr=(0.6)
# declare -a model_id=(1 2 3 4 5 6 7 8 9 10)

# USE THIS FOR L2R
# declare -a arr=("learn2reweight") # L2R in Table2 Column2 (Question)
# declare -a gamma_arr=(0.1) # not actually used
# declare -a lamda_arr=(0.01) # meta-learning rate
# declare -a model_id=(1 2 3 4 5)


# USE THIS FOR Only-L
# declare -a arr=("f_d") # Only-L in Table2 Column2 (Question) 
# declare -a gamma_arr=(0.1) # not actully used
# declare -a lamda_arr=(0.1) # not actully used
# declare -a model_id=(1 2 3 4 5)

EPOCHS=3
LR=0.0001 #not used while testing
DROPOUT_KEEP_PROB=1.0
VALID_PICKLE_NAME=test_processed.p #used test pickle while testing
D_PICKLE_NAME=d_processed.p
U_PICKLE_NAME=U_processed.p

for MODE in "${arr[@]}"
do
   echo "$MODE"
   mode=$MODE   
   if [ "$MODE" == "f_d" ];then
      CKPT_LOAD_MODE=f_d
   else
      CKPT_LOAD_MODE=f_d_U
   fi

   if [[ "$MODE" = "implication" || "$MODE" = "pr_loss" ]];then
      test_mode=test_all # both test f and test w
   else
      test_mode=test_f # only test_f is meaningful because no "w network" in methods other than implication and pr_loss
   fi

   if [[ "$MODE" = "implication" ]];then
      USE_JOINT_f_w=True  #this flag should be True if you need to use output of rule network while doing inference
   else
      USE_JOINT_f_w=False  
   fi

   for GAMMA in "${gamma_arr[@]}"
   do
      for LAMDA in "${lamda_arr[@]}"
      do
         for Q in "${model_id[@]}"
         do
            nohup ./my_TREC.sh "$MODE"_"$GAMMA"_"$LAMDA"_"$Q" $test_mode $EPOCHS $LR \
            $CKPT_LOAD_MODE $DROPOUT_KEEP_PROB $D_PICKLE_NAME \
            $VALID_PICKLE_NAME $U_PICKLE_NAME $GAMMA $LAMDA $USE_JOINT_f_w > $logdir/test_"$MODE"_"$GAMMA"_"$LAMDA"_"$Q".txt &
         done
      done
   done
done
