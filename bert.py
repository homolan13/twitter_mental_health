import numpy as np
import pandas as pd
import torch as th
from tqdm import trange
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

# Necessary pre-processing
def preprocessing(input_text, tokenizer):
    '''
    Soem parts copied from https://towardsdatascience.com/fine-tuning-bert-for-text-classification-54e7df642894
    Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
        - input_ids: list of token ids
        - token_type_ids: list of token type ids
        - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
    '''
    return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens=True,
                        max_length=64,
                        truncation=True,
                        padding='max_length',
                        return_attention_mask=True,
                        return_tensors='pt'
                )


def main():
    raw = pd.read_csv('data/Mental-Health-Twitter.csv', index_col=0)
    data = raw[['post_text', 'label']]

    #Load BERT tokenizer and classifier as well as the optimizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, output_attentions=False, output_hidden_states=False)
    model.cuda()
    optimizer = th.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-08)

    

    token_id = []
    attention_masks = []
    for tweet in data['post_text'].values:
        encoding_dict = preprocessing(tweet, tokenizer)
        token_id.append(encoding_dict['input_ids']) 
        attention_masks.append(encoding_dict['attention_mask'])

    token_id = th.cat(token_id, dim = 0)
    attention_masks = th.cat(attention_masks, dim = 0)
    labels = th.tensor(data['label'].values)

    batch_size = 16

    # Indices of the train and validation splits stratified by labels
    train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.3, shuffle = True, stratify = labels)

    # Train and validation sets
    train_set = TensorDataset(token_id[train_idx], 
                            attention_masks[train_idx], 
                            labels[train_idx])

    val_set = TensorDataset(token_id[val_idx], 
                            attention_masks[val_idx], 
                            labels[val_idx])

    # Prepare DataLoader
    train_dataloader = DataLoader(
                train_set,
                sampler=RandomSampler(train_set),
                batch_size=batch_size
            )

    validation_dataloader = DataLoader(
                val_set,
                sampler=SequentialSampler(val_set),
                batch_size=batch_size
            )

    def b_tp(preds, labels):
        '''Returns True Positives (TP): count of correct predictions of actual class 1'''
        return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

    def b_fp(preds, labels):
        '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
        return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

    def b_tn(preds, labels):
        '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
        return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

    def b_fn(preds, labels):
        '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
        return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

    def b_metrics(preds, labels):
        '''
        Returns the following metrics:
            - accuracy    = (TP + TN) / N
            - precision   = TP / (TP + FP)
            - recall      = TP / (TP + FN)
            - specificity = TN / (TN + FP)
        '''
        preds = np.argmax(preds, axis = 1).flatten()
        labels = labels.flatten()
        tp = b_tp(preds, labels)
        tn = b_tn(preds, labels)
        fp = b_fp(preds, labels)
        fn = b_fn(preds, labels)
        b_accuracy = (tp + tn) / len(labels)
        b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
        b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
        b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
        return b_accuracy, b_precision, b_recall, b_specificity


    epochs = 2

    for _ in trange(epochs, desc = 'Epoch'):
        
        # ========== Training ==========
        
        # Set model to training mode
        model.train()
        
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            # Forward pass
            train_output = model(b_input_ids, 
                                token_type_ids = None, 
                                attention_mask = b_input_mask, 
                                labels = b_labels)
            # Backward pass
            train_output.loss.backward()
            optimizer.step()
            # Update tracking variables
            tr_loss += train_output.loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        # ========== Validation ==========

        # Set model to evaluation mode
        model.eval()

        # Tracking variables 
        val_accuracy = []
        val_precision = []
        val_recall = []
        val_specificity = []

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with th.no_grad():
                # Forward pass
                eval_output = model(b_input_ids, 
                                token_type_ids = None, 
                                attention_mask = b_input_mask)
            logits = eval_output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            # Calculate validation metrics
            b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
            val_accuracy.append(b_accuracy)
            # Update precision only when (tp + fp) !=0; ignore nan
            if b_precision != 'nan': val_precision.append(b_precision)
            # Update recall only when (tp + fn) !=0; ignore nan
            if b_recall != 'nan': val_recall.append(b_recall)
            # Update specificity only when (tn + fp) !=0; ignore nan
            if b_specificity != 'nan': val_specificity.append(b_specificity)

        print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
        print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
        print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
        print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
        print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')



if __name__ == '__main__':
    main()