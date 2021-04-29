stage=0


listsdir='../DiCOVA_Train_Val_Data_Release/LISTS/'
audiodir='../DiCOVA_Train_Val_Data_Release/AUDIO/'
metadatafil='../DiCOVA_Train_Val_Data_Release/metadata.csv'

datadir='data'
feature_dir='feats'
output_dir='results'

train_config='conf/train.conf'
feats_config='conf/feature.conf'

. parse_options.sh

if [ $stage -le 0 ]; then
	echo "==== Preparing data folders ====="
	mkdir -p $datadir
	cat $listsdir/*.txt | sort | uniq >$datadir/allfiles.scp
	awk -v audiodir=$audiodir {'print $1" "audiodir"/"$1".flac"'} < $datadir/allfiles.scp >$datadir/wav.scp
	cat $metadatafil | awk {'split($1,a,",");print a[1]" "a[2]'} >$datadir/labels
	for fold in $(seq 1 5);do
		mkdir -p $datadir/fold_$fold
		for item in train val;do
			awk -v audiodir=$audiodir {'print $1" "audiodir"/"$1".flac"'} < $listsdir/${item}_fold_$fold.txt >$datadir/fold_$fold/$item.scp	
			awk 'NR==FNR{_[$1];next}($1 in _)' $listsdir/${item}_fold_${fold}.txt $datadir/labels >$datadir/fold_$fold/${item}_labels
		done
	done
fi

if [ $stage -le 1 ]; then
	# Creates a separate pickle file containing feature matrix for each recording in the wav.scp
	# Expects a data folder, with train_dev and eval folders inside. each folder has a wav.scp file
	# Each row in wav.scp is formatted as: <wav_id> <wav_file_path>
	# Feature matrices are written to: $feature_dir/{train_dev/eval_set}/<wav_id>_<feature_type>.pkl
	# feature.conf specifies configuration settings for feature extraction

	echo "==== Feature extraction ====="
	mkdir -p $feature_dir
	python feature_extraction.py -c $feats_config -i $datadir/wav.scp -o $feature_dir
	cp $feature_dir/feats.scp $datadir/feats.scp
fi

# Logistic Regression
if [ $stage -le 2 ]; then
	output_dir='results_lr'
	train_config='conf/train_lr.conf'

	mkdir -p $output_dir
	echo "========= Logistic regression classifier ======================"
	cat $train_config
	for fold in $(seq 1 5);do
		mkdir -p $output_dir/fold_${fold}
		cp $datadir/feats.scp $datadir/fold_${fold}/
		# Train
		python train.py	-c $train_config -d $datadir/fold_${fold} -o $output_dir/fold_${fold}
		# Validate
		python infer.py --modelfil $output_dir/fold_${fold}/model.pkl --featsfil $datadir/fold_${fold}/feats.scp --file_list $datadir/fold_${fold}/val.scp --outfil $output_dir/fold_${fold}/val_scores.txt
		# Score
		python scoring.py --ref_file $datadir/fold_${fold}/val_labels --target_file $output_dir/fold_${fold}/val_scores.txt --output_file $output_dir/fold_${fold}/val_results.pkl
	done
	# below file can be uploaded to scoring server to appear on leaderboard for development set performance 
	cat $output_dir/fold_1/val_scores.txt $output_dir/fold_2/val_scores.txt $output_dir/fold_3/val_scores.txt $output_dir/fold_4/val_scores.txt $output_dir/fold_5/val_scores.txt > $output_dir/val_scores_allfolds.txt
	# summarize all folds performance
	python summarize.py $output_dir

fi

# Random Forest 
if [ $stage -le 3 ]; then
	output_dir='results_rf'
	train_config='conf/train_rf.conf'

	mkdir -p $output_dir
	echo "========= Random forest classifier ======================"
	cat $train_config
	for fold in $(seq 1 5);do
		mkdir -p $output_dir/fold_${fold}
		cp $datadir/feats.scp $datadir/fold_${fold}/
		# Train
		python train.py	-c $train_config -d $datadir/fold_${fold} -o $output_dir/fold_${fold}
		# Validate
		python infer.py --modelfil $output_dir/fold_${fold}/model.pkl --featsfil $datadir/fold_${fold}/feats.scp --file_list $datadir/fold_${fold}/val.scp --outfil $output_dir/fold_${fold}/val_scores.txt
		# Score
		python scoring.py --ref_file $datadir/fold_${fold}/val_labels --target_file $output_dir/fold_${fold}/val_scores.txt --output_file $output_dir/fold_${fold}/val_results.pkl 
	done
	# below file can be uploaded to scoring server to appear on leaderboard for development set performance 
	cat $output_dir/fold_1/val_scores.txt $output_dir/fold_2/val_scores.txt $output_dir/fold_3/val_scores.txt $output_dir/fold_4/val_scores.txt $output_dir/fold_5/val_scores.txt > $output_dir/val_scores_allfolds.txt
	# summarize all folds performance
	python summarize.py $output_dir
fi

# Multi-Layer Perceptron
if [ $stage -le 4 ]; then
	output_dir='results_mlp'
	train_config='conf/train_mlp.conf'

	mkdir -p $output_dir
	echo "========= Multilayer perceptron classifier ======================"
	cat $train_config
	for fold in $(seq 1 5);do
		mkdir -p $output_dir/fold_${fold}
		cp $datadir/feats.scp $datadir/fold_${fold}/
		# Train
		python train.py	-c $train_config -d $datadir/fold_${fold} -o $output_dir/fold_${fold}
		# Validate
		python infer.py --modelfil $output_dir/fold_${fold}/model.pkl --featsfil $datadir/fold_${fold}/feats.scp --file_list $datadir/fold_${fold}/val.scp --outfil $output_dir/fold_${fold}/val_scores.txt
		# Score
		python scoring.py --ref_file $datadir/fold_${fold}/val_labels --target_file $output_dir/fold_${fold}/val_scores.txt --output_file $output_dir/fold_${fold}/val_results.pkl 
	done
	# below file can be uploaded to scoring server to appear on leaderboard for development set performance 
	cat $output_dir/fold_1/val_scores.txt $output_dir/fold_2/val_scores.txt $output_dir/fold_3/val_scores.txt $output_dir/fold_4/val_scores.txt $output_dir/fold_5/val_scores.txt > $output_dir/val_scores_allfolds.txt
	# summarize all folds performance
	python summarize.py $output_dir

fi

echo "Done!!!"
