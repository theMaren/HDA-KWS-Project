# Human data analytics: Key Word Spotting
This repository showcases our project for the Human Data Analytics course at the University of Padova, academic year 2023/24. Focused on keyword spotting, a core task in speech recognition used for example in voice assistants like Amazon Alexa. We used the [Google Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands) for our research. We explored various audio preprocessing techniques and neural network architectures, evaluating their performance and ressource consumption.

## Remarks
Our code was executed on Google Colab, utilizing the T4 GPU resource available on the platform. Therefore, when running the code in a different environment, runtime and memory consumption may vary. Additionally, file path settings should be adjusted to fit the structure of the user's environment. We attempted to ensure reproducibility of our results by setting random seeds as follows:

- random.seed(42)
- np.random.seed(42)
- tf.random.set_seed(42)

However, despite these efforts, we observed variations in the results with each execution of our code and during model training, therefore we discarded the approach. Consequently, it should be anticipated that different outcomes may be obtained when running the code.

## Notebooks

### complete_preprocessing.ipynb
This notebook presents an exploratory analysis of the dataset and details our preprocessing pipeline. It covers various audio data transformations into images and data downsampling techniques.

### Baseline_model.ipynb
This notebook outlines our baseline model, against which our developed models are benchmarked. The model design follows guidelines from the [TensorFlow audio recognition tutorial](https://www.tensorflow.org/tutorials/audio/simple_audio).

### Conformer_model.ipynb
This notebook demonstrates the application of a compact conformer model to our dataset. The conformer model adheres to the architecture outlined in the literature. 

### Conformer_ensemble_model.ipynb
This notebook attempts to enhance the Conformer model by constructing an ensemble of three distinct, smaller Conformer models

## Python scripts

### train_test_val_split.py
The script divides the dataset into train, validation, and test sets according to the split suggested by the dataset's creator. For each set, a separate folder is created. Within each folder, the structure of the original dataset is maintained, with one subfolder for each keyword.

### preprocessing_tf.py and preprocessing.py
Both scripts contain identical functions used for preprocessing. The key difference lies in the data format: preprocessing.py processes data as numpy arrays, while preprocessing_tf.py handles data as tensors. The approach used in preprocessing.py is intended for exploratory data analysis and demonstrating the preprocessing pipeline. Conversely, preprocessing_tf.py is employed during model development since it already operates with tensors, ensuring the data is in the appropriate format for training models.

### conformer.py
The script includes the classes for constructing our conformer model and performing hyperparameter optimization for it.

### evaluation.py
The script includes methods for evaluating models based on performance, time, and resource consumption.

### demo.py
Running this script initiates the demo. The demo is designed so that a model and an audio input device must be selected. Afterwards, the user can speak a keyword into the microphone, and the prediction from the selected model will be returned.

## Models
The folder contains all of our trained models, which have been saved in TensorFlow format following the completion of training.
