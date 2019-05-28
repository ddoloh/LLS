echo train binary_LBP=0, train binary=1, test_binary=2, train_binary_LBP=3, testing lbp_lle=4, train binary_clf_CLBP=5, train binary_clf_UCLPB=6, train binary_clf_LBP_llesmote=7, train binary_clf_CLBP_llesmote=8
read value
if [ ${value} -eq 0 ];then
CUDA_VISIBLE_DEVICES=0 python3 gru_svm_main.py --operation "train" \
--checkpoint_path models/checkpoint/binary_clf_LBP \
--model_name gru_svm \
--log_path models/logs/binary_clf_LBP \
--result_path results/binary_clf_LBP \
--LEARNING_RATE 0.0007 \
--experimental_method binary_clf_LBP \
--n_neighbor 3
fi

if [ ${value} -eq 1 ];then
CUDA_VISIBLE_DEVICES=1 python3 gru_svm_main.py --operation "train" \
--checkpoint_path models/checkpoint/binary_clf \
--model_name gru_svm \
--log_path models/logs/binary_clf \
--result_path results/binary_clf \
--LEARNING_RATE 0.0007 \
--experimental_method binary_clf \
--n_neighbor 3
fi

if [ ${value} -eq 2 ];then
CUDA_VISIBLE_DEVICES=1 python3 gru_svm_main.py --operation "test" \
--checkpoint_path models/checkpoint/binary_clf \
--result_path results/binary_clf_LBP \
--experimental_method binary_clf \
--n_neighbor 3
fi

if [ ${value} -eq 3 ];then
CUDA_VISIBLE_DEVICES=1 python3 gru_svm_main.py --operation "train" \
--checkpoint_path models/checkpoint/binary_clf_lbplle \
--model_name gru_svm_llesmote \
--log_path models/logs/binary_clf_lbplle \
--result_path results/binary_clf_lbplle \
--LEARNING_RATE 0.0004 \
--experimental_method binary_clf_LBP \
--n_neighbor 3
fi

if [ ${value} -eq 4 ];then
CUDA_VISIBLE_DEVICES=1 python3 gru_svm_main.py --operation "test" \
--checkpoint_path models/checkpoint/binary_clf_lbplle \
--result_path results/binary_clf_lbplle \
--experimental_method binary_clf_LBP \
--n_neighbor 3
fi

if [ ${value} -eq 5 ];then
CUDA_VISIBLE_DEVICES=1 python3 gru_svm_main.py --operation "train" \
--checkpoint_path models/checkpoint/binary_clf_CLBP \
--model_name gru_svm \
--log_path models/logs/binary_clf_CLBP \
--result_path results/binary_clf_CLBP \
--LEARNING_RATE 0.0004 \
--experimental_method binary_clf_CLBP \
--n_neighbor 3
fi

if [ ${value} -eq 6 ];then
CUDA_VISIBLE_DEVICES=0 python3 gru_svm_main.py --operation "train" \
--checkpoint_path models/checkpoint/binary_clf_UCLBP \
--model_name gru_svm \
--log_path models/logs/binary_clf_UCLBP \
--result_path results/binary_clf_UCLBP \
--LEARNING_RATE 0.0004 \
--experimental_method binary_clf_UCLBP \
--n_neighbor 3
fi

if [ ${value} -eq 7 ];then
CUDA_VISIBLE_DEVICES=0 python3 gru_svm_main.py --operation "train" \
--checkpoint_path models/checkpoint/binary_clf_LBP_llesmote \
--model_name gru_svm_llesmote \
--log_path models/logs/binary_clf_LBP_llesmote \
--result_path results/binary_clf_LBP_llesmote \
--experimental_method binary_clf_LBP \
--n_neighbor 3
fi

if [ ${value} -eq 8 ];then
CUDA_VISIBLE_DEVICES=1 python3 gru_svm_main.py --operation "train" \
--checkpoint_path models/checkpoint/binary_clf_CLBP_llesmote \
--model_name gru_svm_llesmote \
--log_path models/logs/binary_clf_CLBP_llesmote \
--result_path results/binary_clf_CLBP_llesmote \
--experimental_method binary_clf_CLBP \
--n_neighbor 3
fi
