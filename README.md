1. Python Environment

	conda create --name my_env python=3.7.13

2. Dependencies installation using iDCWE_ENV.yml file:
	
	conda activate my_env
	
	INSTALL ALL THE 4 .whl FILES:
	
	pip install _____.whl
	
	conda env update --name my_env --file iDCWE_ENV.yml 
	
	

3. Stopwords download:

	run the following command in the environment

	python -m nltk.downloader stopwords

4. Download Dataset:
	
	link for ciao raw dataset: https://www.cse.msu.edu/~tangjili/trust
	
	link for arxiv dataset : https://www.kaggle.com/Cornell-University/arxiv
	
	link for yelp dataset: https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset (Download yelp_academic_dataset_user.json & yelp_academic_dataset_review.json files)
	
	link for reddit dataset: https://www.kaggle.com/datasets/kaggle/reddit-comments-may-2015

5. Preprocess the data:

	in 'data' folder, run the following command (for ciao dataset)

		python preprocessing_data_ciao.py

6. Train Vectors:

	in the 'data' folder, run the following command 

		python train_vectors.py --dim 50

7. Pickle Data:

	in the main folder, run the following commands 

		python src/pickle_data.py --data_dir data --data ciao --task mlm --social_dim 50
		python src/pickle_data.py --data_dir data --data ciao --task sa --social_dim 50

8. Uploading results:

	step 7 should create the following files in the 'data' folder 
		data/mlm_ciao_50_train.p
		data/mlm_ciao_50_test.p
		data/mlm_ciao_50_dev.p
		data/sa_ciao_50_train.p
		data/sa_ciao_50_test.p
		data/sa_ciao_50_dev.p


		
9. Running the iDCWE mlm_model:
	
	in the main folder, run the following command
	
		python src/idcwe_main_mlm.py --data_dir data --results_dir results --trained_dir trained --data ciao --device 0 --lr 0.000003 --n_epochs 7 --lambda_a 0.1 --lambda_w 100 --social_dim 50 --gnn roland --save_interval 5 --mini_batch_size 15686
		
		python src/idcwe_main_mlm.py --data_dir data --results_dir results --trained_dir trained --data yelp --device 1 --lr 0.000003 --n_epochs 7 --lambda_a 0.1 --lambda_w 100 --social_dim 50 --gnn roland --save_interval 5 --mini_batch_size 34789
		
		
		
10. Running the iDCWE sa_model:

	In the main folder, run the following command
	
		python src/idcwe_main_sa.py --data_dir data --results_dir results --trained_dir trained --data ciao --device 0 --lr 0.000003 --n_epochs 7 --lambda_a 0.1 --lambda_w 100 --social_dim 50 --gnn roland --save_interval 5 --mini_batch_size 15686
		
		python src/idcwe_main_sa.py --data_dir data --results_dir results --trained_dir trained --data yelp --device 1 --lr 0.000003 --n_epochs 7 --lambda_a 0.1 --lambda_w 100 --social_dim 50 --gnn roland --save_interval 5 --mini_batch_size 34789
	
