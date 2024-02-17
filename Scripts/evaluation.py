from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,accuracy_score
import tensorflow as tf
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import subprocess
import psutil


#get the cpu usage (for google colab environment)
def get_system_ram_usage():
    ram_info = psutil.virtual_memory()
    used_ram = ram_info.used / (1024 ** 3)# Convert bytes to GB

    return used_ram

#log the gpu usage every interval seconds (for google colab environment)
def log_gpu_usage(log_file_path, stop_event, interval):
    with open(log_file_path, 'a') as log_file:
        while not stop_event.is_set():
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
            log_file.write(result.stdout)
            log_file.write('\n' + '-'*80 + '\n')
            log_file.flush()
            time.sleep(interval)


#read the gpu usage from the log file, plot it and put metrics tp dataframe
def get_gpu_usage(logfile_path):

    with open(logfile_path, 'r') as file:
        log_data = file.read()

    #extract gpu pattern from logfile
    gpu_util_pattern = re.compile(r'\|\s*(\d+)%\s+Default')
    memory_usage_pattern = re.compile(r'\|\s+(\d+)MiB / (\d+)MiB')

    gpu_utils = gpu_util_pattern.findall(log_data)
    memory_usages = memory_usage_pattern.findall(log_data)

    df = pd.DataFrame({
        'GPU Utilization (%)': [int(util) for util in gpu_utils],
        'Memory Usage (MiB)': [int(usage[0]) for usage in memory_usages],
        'Memory Usage %': [round((int(usage[0]) / int(usage[1])) * 100, 3) for usage in memory_usages]
    })

    plt.plot(df.index, df['Memory Usage (MiB)'], marker='x', color='darkred')
    plt.title('GPU Memory Usage (MIB)')
    plt.xlabel('Time (Arbitrary Units)')
    plt.ylabel('Memory Usage (MIB)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return df

#return dataframe of common error metrics
def get_error_metrics(model_name, true_labels, y_pred_logits):

    predicted_classes = tf.argmax(y_pred_logits, axis=1)

    precision = precision_score(true_labels, predicted_classes, average='macro')
    recall = recall_score(true_labels, predicted_classes, average='macro')
    f1 = f1_score(true_labels, predicted_classes, average='macro')
    accuracy = accuracy_score(true_labels, predicted_classes)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    cross_entropy_loss = loss_fn(true_labels, y_pred_logits).numpy()

    metrics_data = {
        'Model Name': [model_name],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1],
        'Accuracy': [accuracy],
        'Cross-Entropy Loss': [cross_entropy_loss]
    }

    # Create the DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    return metrics_df

#return the class name for numeric label
def numeric_to_string_labels(vocabulary, numeric_labels):
    return np.array(vocabulary)[numeric_labels]

#plot confusion matrix for a given prediction and true labels combination
def plot_confusion_matrix(true_labels, y_pred_logits,label_lookup):

    predicted_classes = tf.argmax(y_pred_logits, axis=1)

    vocabulary = label_lookup.get_vocabulary()
    true_labels_strings = numeric_to_string_labels(vocabulary, true_labels)
    predicted_classes_strings = numeric_to_string_labels(vocabulary, predicted_classes)

    unique_labels_strings = np.unique(np.concatenate([true_labels_strings, predicted_classes_strings]))

    conf_matrix = confusion_matrix(true_labels_strings, predicted_classes_strings, labels=unique_labels_strings)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels_strings,
                yticklabels=unique_labels_strings)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.show()


#plot the training history of a model
def plot_fit_history(history, loss_columns, accuracy_columns):
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    for column in loss_columns:
        plt.plot(history.history[column], label=column)
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    for column in accuracy_columns:
        plt.plot(history.history[column], label=column)
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()