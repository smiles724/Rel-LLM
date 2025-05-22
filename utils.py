import numpy as np
from torch.nn import BCEWithLogitsLoss, L1Loss
import torch.nn as nn
import torch.nn.init as init

from relbench.base import TaskType

rel_info = {'rel-event': ['user-attendance', 'user-ignore', 'user-repeat'], 'rel-amazon': ['user-churn', 'item-churn', 'user-ltv', 'item-ltv'],
            'rel-hm': ['user-churn', 'item-sales'], 'rel-stack': ['post-votes', 'user-engagement', 'user-badge'], 'rel-trial': ['study-outcome', 'study-adverse', 'site-success'],
            'rel-f1': ['driver-position', 'driver-dnf', 'driver-top3'], 'rel-avito': ['user-visits', 'user-clicks', 'ad-ctr']}

description_dict = {'rel-event': {'user-attendance': 'This task is to predict how many events each user will respond yes or maybe in the next seven days.',
                                  'user-ignore': 'This task is to predict whether a user will ignore more than 2 event invitations in the next 7 days.',
                                  'user-repeat': 'This task is to predict whether a user will attend an event (by responding yes or maybe) in the next 7 days if '
                                                 'they have already attended an event in the last 14 days.'},
                    'rel-amazon': {'user-churn': 'This task is to predict if the customer will review any product in the next 3 months or not.',
                                   'item-churn': 'This task is to predict if the product will receive any reviews in the next 3 months or not.',
                                   'user-ltv': 'This task is to predict the $ value of the total number of products each user will buy and review in the next 3 months.',
                                   'item-ltv': 'This task is to predict the $ value of the total number purchases and reviews each product will receive in the next 3 months.'},
                    'rel-stack': {'post-votes': 'This task is to predict how many votes this user post will receive in the next 3 months.',
                                  'user-engagement': 'This task is to predict if a user will make any votes, posts, or comments in the next 3 months or not.',
                                  'user-badge': 'This task is to predict if a user will receive a new badge in the next 3 months or not.'},
                    'rel-avito': {'user-visits': 'This task is to predict whether this customer will visit more than one Ad in the next 4 days or not.',
                                  'user-clicks': 'This task is to predict whether this customer will click on more than one Ads in the next 4 day or not.',
                                  'ad-ctr': 'Assuming the Ad will be clicked in the next 4 days, this task is to predict the Click-Through-Rate (CTR) for each Ad.'},
                    'rel-f1': {'driver-position': 'This task is to predict the average finishing position of each driver all races in the next 2 months.',
                               'driver-dnf': 'This task is to predict if this driver will finish a race in the next 1 month or not.',
                               'driver-top3': 'This task is to predict if this driver will qualify in the top-3 for a race in the next 1 month or not.'},
                    'rel-trial': {'study-outcome': 'This task is to predict if the trial in the next 1 year will achieve its primary outcome or not.',
                                  'study-adverse': 'This task is to predict the number of affected patients with severe adverse events/death for the trial in the next 1 year.',
                                  'site-success': 'This task is to predict the success rate of a trial site in the next 1 year.'},
                    'rel-hm': {'item-sales': 'This task is to predict the total sales for an article in the next week.',
                               'user-churn': 'This task is to predict the churn for a customer (no transactions) in the next week.'}}
question_dict = {'rel-event': {'user-attendance': 'What is the attendance of user? Give an integer as an answer.',
                               'user-ignore': 'Given recent activity and event history, will this user ignore more than 2 event invitations in the next 7 days? Give Yes or No as an answer.',
                               'user-repeat': 'Given recent activity and event history, will this user attend an event in the next 7 days? Give Yes or No as an answer.'},
                 'rel-amazon': {'user-churn': 'Based on the customer data provided, will this customer review any product in the next 3 months? Give Yes or No as an answer.',
                                'item-churn': 'Based on the product data provided, will the product receive any reviews in the next 3 months? Give Yes or No as an answer.',
                                'user-ltv': 'What is the total dollar value of products this user will buy and review in the next 3 months? Provide a float numerical answer.',
                                'item-ltv': 'What is the total dollar value of purchases this product will receive in the next 3 months? Provide a float numerical answer.'},
                 'rel-stack': {'post-votes': 'Based on records of activity, how many votes will this user post receive in the next 3 months? Give an integer as an answer.',
                               'user-engagement': 'Based on records of activity, will this user make any votes, posts, or comments in the next 3 months? Give Yes or No as an answer.',
                               'user-badge': 'Based on records of activity, will this user receive a new badge in the next 3 months? Give Yes or No as an answer.'},
                 'rel-avito': {'user-visits': 'Will this customer visit more than one Ad in the next 4 days? Give Yes or No as an answer.',
                               'user-clicks': 'Will this customer click on more than one Ads in the next 4 day? Give Yes or No as an answer.',
                               'ad-ctr': 'What is the Click-Through-Rate (CTR) for this Ad?'},
                 'rel-f1': {'driver-position': 'What is the average finishing position of this driver all races in the next 2 months? Provide a float numerical answer.',
                            'driver-dnf': 'Will this driver finish a race in the next 1 month? Give Yes or No as an answer.',
                            'driver-top3': 'Will this driver qualify in the top-3 for a race in the next 1 month? Give Yes or No as an answer.'},
                 'rel-hm': {'item-sales': 'What is the sum of prices of the associated transactions?', 'user-churn': 'Will this customer make no transactions in the next week?'},
                 'rel-trial': {'study-outcome': 'Will the trial achieve its primary outcome in the next 1 year?',
                               'study-adverse': 'What is the number of affected patients with severe adverse events/death for the trial in the next 1 year? Give an integer as an answer.',
                               'site-success': 'What is the success rate of a trial site in the next 1 year?'}}

# TODO: check ratio of 1/0 in the dataset
def task_info(task):
    clamp_min, clamp_max = None, None
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        loss_fn = BCEWithLogitsLoss()
        tune_metric = "roc_auc"
        higher_is_better = True
    elif task.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fn = L1Loss()
        tune_metric = "mae"
        higher_is_better = False
        # Get the clamp value at inference time
        train_table = task.get_table("train")
        clamp_min, clamp_max = np.percentile(train_table.df[task.target_col].to_numpy(), [2, 98])
    elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        out_channels = task.num_labels
        loss_fn = BCEWithLogitsLoss()
        tune_metric = "multilabel_auprc_macro"
        higher_is_better = True
    else:
        raise ValueError(f"Task type {task.task_type} is unsupported")
    return out_channels, loss_fn, tune_metric, higher_is_better, clamp_min, clamp_max


# Function to reset and initialize the parameters
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        # Initialize weights using Kaiming Uniform (or any desired method)
        init.kaiming_uniform_(module.weight, nonlinearity='linear')
        # Initialize biases to zero
        if module.bias is not None:
            init.zeros_(module.bias)
