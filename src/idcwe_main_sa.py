import argparse
import logging
import pickle
import random
import time
import math
import torch
from torch import optim,nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import os
from data_helpers import *
from model import SAModel
from tqdm import tqdm
import networkx as nx
from node2vec import Node2Vec


logging.disable(logging.WARNING)

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=None, type=str, required=True, help='Data directory.')
parser.add_argument('--results_dir', default=None, type=str, required=True, help='Results directory.')
parser.add_argument('--trained_dir', default=None, type=str, required=True, help='Trained model directory.')
parser.add_argument('--mini_batch_size', default=None, type=int, required=True, help='Mini Batch size.')
parser.add_argument('--lr', default=None, type=float, required=True, help='Learning rate.')
parser.add_argument('--n_epochs', default=None, type=int, required=True, help='Number of epochs to train the mini-batch.')
parser.add_argument('--lambda_a', default=0, type=float, required=True, help='Regularization constant a.')
parser.add_argument('--lambda_w', default=0, type=float, required=True, help='Regularization constant w.')
parser.add_argument('--device', default=None, type=int, required=True, help='Selected CUDA device.')
parser.add_argument('--data', default=None, type=str, required=True, help='Name of data.')
parser.add_argument('--social_dim', default=50, type=int, help='Size of social embeddings.')
parser.add_argument('--gnn', default=None, type=str, help='Type of graph neural network.')
parser.add_argument('--social_only', default=False, action='store_true', help='Only use social information.')
parser.add_argument('--time_only', default=False, action='store_true', help='Only use temporal information.')
parser.add_argument('--save_interval', default=None, type=int, required=True, help='Number of epochs to checkpoint model')
args = parser.parse_args()



with open('{}/{}_edges.p'.format(args.data_dir, args.data), 'rb') as f:
    edge_set = pickle.load(f)
with open('{}/{}_users.p'.format(args.data_dir, args.data), 'rb') as f:
    users = pickle.load(f)

print('Load training data...')
with open('{}/sa_{}_{}_train.p'.format(args.data_dir, args.data, args.social_dim), 'rb') as f:
    train_dataset = pickle.load(f)
print('Load development data...')
with open('{}/sa_{}_{}_dev.p'.format(args.data_dir, args.data, args.social_dim), 'rb') as f:
    dev_dataset = pickle.load(f)
print('Load test data...')
with open('{}/sa_{}_{}_test.p'.format(args.data_dir, args.data, args.social_dim), 'rb') as f:
    test_dataset = pickle.load(f)

print('Lambda a: {:.0e}'.format(args.lambda_a))
print('Lambda w: {:.0e}'.format(args.lambda_w))
print('Social embeddings dimensionality: {}'.format(args.social_dim))
print('Number of time units: {}'.format(train_dataset.n_times))
print('Number of vocabulary items: {}'.format(len(train_dataset.filter_tensor)))

collator = SACollator(train_dataset.user2id)
dev_loader = DataLoader(dev_dataset, batch_size=4, collate_fn=collator)
test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collator)

filename = 'sa_{}'.format(args.data)
filename += '_{}'.format(args.social_dim)
if args.social_only:
    filename += '_s'
elif args.time_only:
    filename += '_t'

device = torch.device('cuda:{}'.format(args.device))
graph_data = train_dataset.graph_data.to(device)
vocab_filter = train_dataset.filter_tensor.to(device)

checkpoint_dir = 'checkpoints'
checkpoint_file = 'sa_model_checkpoint.pt'
best_perplexity = None

# best_result = get_best('{}/{}.txt'.format(args.results_dir, filename), metric='perplexity')
# if best_result:
#     best_perplexity = best_result[0]
# else:
#     best_perplexity = None
# print('Best perplexity so far: {}'.format(best_perplexity))

print('Train model...')

train_df = pd.read_csv('{}/{}_train.csv'.format(args.data_dir,args.data))

# Set the batch size and initial row range
mini_batch_size = args.mini_batch_size
start_row = 0
prev_samples = pd.DataFrame()  # Create an empty dataframe to store previous samples

while start_row < len(train_df):

    start_time = time.time()


    flag = True
    
    # Get the end row for the current batch
    end_row = min(start_row + mini_batch_size, len(train_df))
    
    # Get the end row for the current batch
    end_row = min(start_row + mini_batch_size, len(train_df))
    
    # Get a random sample of 20% from all previous batches
    prev_sample_size = int(len(prev_samples) * 0.2)
    prev_sample = prev_samples.sample(n=prev_sample_size, random_state=1)
    
    # Merge the sample into the current batch
    current_batch = pd.concat([prev_sample, train_df.loc[start_row:end_row]])
    #print(len(current_batch))
    current_batch.to_csv('combined_data.csv',index=False)

    users1 = set(current_batch.user)
    user_list = list(users1)
    first_n_users = user_list
    
    # Filter edge set using the first n users
    filtered_edges = set([edge for edge in edge_set if edge[0] in first_n_users and edge[1] in first_n_users])
    
    # Filter user set using the first n users
    filtered_users = set(user_list)
    
    graph = nx.Graph()
    graph.add_nodes_from(filtered_users)
    graph.add_edges_from(filtered_edges)
    
    
    n2v = Node2Vec(graph, dimensions=50, walk_length=80, num_walks=10, workers=1)
    n2v_model = n2v.fit(window=2, min_count=1, epochs=10, seed=123)
    
    n2v_model.wv.save_word2vec_format('{}/ciao_train_vectors_{}.txt'.format(args.data_dir,50), binary=False)
    
    dataset_split_train = SADataset(args.data, split='train', social_dim=50, data_dir=args.data_dir)
    collator1 = SACollator(dataset_split_train.user2id)
    
    train_loader1 = DataLoader(dataset_split_train, collate_fn=collator1, shuffle=True)
    
    graph_data_train = dataset_split_train.graph_data.to(device)
    vocab_filter_train = dataset_split_train.filter_tensor.to(device)
    

    
    if os.path.exists(checkpoint_dir):
        model = SAModel(
        n_times=train_dataset.n_times,
        social_dim=50,
        gnn= args.gnn)
        optimizer = optim.Adam(model.parameters(), lr=args.lr) 
        criterion = nn.BCELoss()
        model.train()
        
        checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        latest_checkpoint_file = max(checkpoint_files, key=os.path.getctime)
        model.load_state_dict(torch.load(latest_checkpoint_file))
        print('\nLatest model loaded: {}\n'.format(latest_checkpoint_file))
        
    else:
        model = SAModel(
        n_times=train_dataset.n_times,
        social_dim=50,
        gnn=args.gnn)
        optimizer = optim.Adam(model.parameters(), lr=args.lr) 
        criterion = nn.BCELoss()
        model.train()
        
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    model = model.to(device)  
    
    for epoch in range(1, args.n_epochs + 1):
        
        print('Epoch : {}'.format(epoch))
 
        model.train()
        
        for i, batch in enumerate(train_loader1):
        
    
            labels, users, times, years, months, days, reviews, masks, segs = batch
    
            labels = labels.to(device)
            times = times.to(device)
            users = users.to(device)
            reviews = reviews.to(device)
            masks = masks.to(device)
            segs = segs.to(device)
    
            optimizer.zero_grad()
    
            offset_t0, offset_t1, output = model(reviews, masks, segs, users, graph_data_train, times, vocab_filter_train)
    
            loss = criterion(output, labels)
            loss += args.lambda_a * torch.norm(offset_t1, dim=-1).pow(2).mean()
            loss += args.lambda_w * torch.norm(offset_t1 - offset_t0, dim=-1).pow(2).mean()
    
            loss.backward()
            
            optimizer.step()
            
           
        
            
        if flag == True and epoch % (args.save_interval) == 0:
            # Save the model
            print('Saving model')
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'{checkpoint_file}.pt'))
	
            flag = False
    
    print('Evaluating model...')
    model.eval()
    
    y_true = list()
    y_pred = list()

    with torch.no_grad():

        for batch in dev_loader:

            labels, users, times, years, months, days, reviews, masks, segs = batch

            labels = labels.to(device)
            users = users.to(device)
            times = times.to(device)
            reviews = reviews.to(device)
            masks = masks.to(device)
            segs = segs.to(device)

            offset_t0, offset_t1, output = model(reviews, masks, segs, users, graph_data, times, vocab_filter)
            
            y_true.extend(labels.tolist())
            y_pred.extend(torch.round(output).tolist())

    f1_dev = f1_score(y_true, y_pred, average='macro')

    y_true = list()
    y_pred = list()

    with torch.no_grad():

        for batch in test_loader:

            labels, users, times, years, months, days, reviews, masks, segs = batch

            labels = labels.to(device)
            times = times.to(device)
            users = users.to(device)
            reviews = reviews.to(device)
            masks = masks.to(device)
            segs = segs.to(device)

            offset_t0, offset_t1, output = model(reviews, masks, segs, users, graph_data, times, vocab_filter)
            
            y_true.extend(labels.tolist())
            y_pred.extend(torch.round(output).tolist())

    f1_test = f1_score(y_true, y_pred, average='macro')
    
    
    # Save the results
    elapsed_time = (time.time() - start_time)/60
    with open('{}/{}.txt'.format(args.results_dir, filename), 'a') as f:
        f.write('f1_dev {:.3f}, f1_test {:.3f}, Elapsed time: {:.2f} mins\n'.format(f1_dev,f1_test,elapsed_time))
        
    
    # Print the results
    print('f1_dev {:.3f}, f1_test {:.3f}, Elapsed time: {:.2f} mins'.format(f1_dev,f1_test,elapsed_time))

   
            
    # Update the start row for the next iteration
    start_row = end_row
    print('Processed {}/{}\n'.format(start_row,len(train_df)))
    
    
    print('-'*100)
    # Add the current batch to the previous samples dataframe
    prev_samples = pd.concat([prev_samples, train_df.loc[start_row-mini_batch_size:end_row-1]])
            


