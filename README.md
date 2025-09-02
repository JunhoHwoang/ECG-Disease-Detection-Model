# PhysioNet Challenge 2025

## Description:

This submission contains the CNN-BiLSTM + Attention model with focal loss. This model utilizes bandpass filter(0.67 - 45 Hz), Z-score normalization, and 7.3 sec center cropping to process ECG signals.

## How to run:

1. Install the dependencies for these scripts by creating a Docker image (see below) or [virtual environment](https://docs.python.org/3/library/venv.html) and running

    pip install -r requirements.txt

2. Train your model by running

    python train_model.py -d training_data -m model

where

- `training_data` (input; required) is a folder with the training data files, which must include the labels; and
- `model` (output; required) is a folder for saving your model.

3. Run your trained model by running

    python run_model.py -d holdout_data -m model -o holdout_outputs

where

- `holdout_data` (input; required) is a folder with the holdout data files, which will not necessarily include the labels;
- `model` (input; required) is a folder for loading your model; and
- `holdout_outputs` (output; required) is a folder for saving your model outputs.

The [Challenge website](https://physionetchallenges.org/2025/#data) provides a training database with a description of the contents and structure of the data files.

4. Evaluate your model by pulling or downloading the [evaluation code](https://github.com/physionetchallenges/evaluation-2025) and running

    python evaluate_model.py -d holdout_data -o holdout_outputs -s scores.csv

where

- `holdout_data`(input; required) is a folder with labels for the holdout data files, which must include the labels;
- `holdout_outputs` (input; required) is a folder containing files with your model's outputs for the data; and
- `scores.csv` (output; optional) is file with a collection of scores for your model.

You can use the provided training set for the `training_data` and `holdout_data` files, but we will use different datasets for the validation and test sets, and we will not provide the labels to your code.

## How do I create data for these scripts?

You can use the scripts in this repository to convert the [CODE-15% dataset](https://zenodo.org/records/4916206), the [SaMi-Trop dataset](https://zenodo.org/records/4905618), and the [PTB-XL dataset](https://physionet.org/content/ptb-xl/) to [WFDB](https://wfdb.io/) format.

Please see the [data](https://physionetchallenges.org/2025/#data) section of the website for more information about the Challenge data.

#### CODE-15% dataset

These instructions use `code15_input` as the path for the input data files and `code15_output` for the output data files, but you can replace them with the absolute or relative paths for the files on your machine.

1. Download and unzip one or more of the `exam_part` files and the `exams.csv` file in the [CODE-15% dataset](https://zenodo.org/records/4916206).

2. Download and unzip the Chagas labels, i.e., the [`code15_chagas_labels.csv`](https://physionetchallenges.org/2025/data/code15_chagas_labels.zip) file.

3. Convert the CODE-15% dataset to WFDB format, with the available demographics information and Chagas labels in the WFDB header file, by running

        python prepare_code15_data.py \
            -i code15_input/exams_part0.hdf5 code15_input/exams_part1.hdf5 \
            -d code15_input/exams.csv \
            -l code15_input/code15_chagas_labels.csv \
            -o code15_output/exams_part0 code15_output/exams_part1

Each `exam_part` file in the [CODE-15% dataset](https://zenodo.org/records/4916206) contains approximately 20,000 ECG recordings. You can include more or fewer of these files to increase or decrease the number of ECG recordings, respectively. You may want to start with fewer ECG recordings to debug your code.

#### SaMi-Trop dataset

These instructions use `samitrop_input` as the path for the input data files and `samitrop_output` for the output data files, but you can replace them with the absolute or relative paths for the files on your machine.

1. Download and unzip `exams.zip` file and the `exams.csv` file in the [SaMi-Trop dataset](https://zenodo.org/records/4905618).

2. Convert the SaMi-Trop dataset to WFDB format, with the available demographics information and Chagas labels in the WFDB header file, by running

        python prepare_samitrop_data.py \
            -i samitrop_input/exams.hdf5 \
            -d samitrop_input/exams.csv \
            -o samitrop_output

#### PTB-XL dataset

These instructions use `ptbxl_input` as the path for the input data files and `ptbxl_output` for the output data files, but you can replace them with the absolute or relative paths for the files on your machine. We are using the `records500` folder, which has a 500Hz sampling frequency, but you can also try the `records100` folder, which has a 100Hz sampling frequency.

1. Download and, if necessary, unzip the [PTB-XL dataset](https://physionet.org/content/ptb-xl/).

2. Update the WFDB files with the available demographics information and Chagas labels by running

        python prepare_ptbxl_data.py \
            -i ptbxl_input/records500/ \
            -d ptbxl_input/ptbxl_database.csv \
            -o ptbxl_output

## How do I run these scripts in Docker?

Docker and similar platforms allow you to containerize and package your code with specific dependencies so that your code can be reliably run in other computational environments.

To increase the likelihood that we can run your code, please [install](https://docs.docker.com/get-docker/) Docker, build a Docker image from your code, and run it on the training data. To quickly check your code for bugs, you may want to run it on a small subset of the training data, such as 1000 records.

If you have trouble running your code, then please try the follow steps to run the example code.

1. Create a folder `example` in your home directory with several subfolders.

        user@computer:~$ cd ~/
        user@computer:~$ mkdir example
        user@computer:~$ cd example
        user@computer:~/example$ mkdir training_data holdout_data model holdout_outputs

2. Download the training data from the [Challenge website](https://physionetchallenges.org/2025/#data). Put some of the training data in `training_data` and `holdout_data`. You can use some of the training data to check your code (and you should perform cross-validation on the training data to evaluate your algorithm).

3. Download or clone this repository in your terminal.

        user@computer:~/example$ git clone https://github.com/physionetchallenges/python-example-2025.git

4. Build a Docker image and run the example code in your terminal.

        user@computer:~/example$ ls
        holdout_data  holdout_outputs  model  python-example-2025  training_data

        user@computer:~/example$ cd python-example-2025/

        user@computer:~/example/python-example-2025$ docker build -t image .

        Sending build context to Docker daemon  [...]kB
        [...]
        Successfully tagged image:latest

        user@computer:~/example/python-example-2025$ docker run -it -v ~/example/model:/challenge/model -v ~/example/holdout_data:/challenge/holdout_data -v ~/example/holdout_outputs:/challenge/holdout_outputs -v ~/example/training_data:/challenge/training_data image bash

        root@[...]:/challenge# ls
            Dockerfile             holdout_outputs        run_model.py
            evaluate_model.py      LICENSE                training_data
            helper_code.py         README.md      
            holdout_data           requirements.txt

        root@[...]:/challenge# python train_model.py -d training_data -m model -v

        root@[...]:/challenge# python run_model.py -d holdout_data -m model -o holdout_outputs -v

        root@[...]:/challenge# python evaluate_model.py -d holdout_data -o holdout_outputs
        [...]

        root@[...]:/challenge# exit
        Exit
