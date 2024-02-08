import os
import shutil

def create_directory_structure(file_list, source_folder, destination_folder):
    with open(file_list, 'r') as f:
        filenames = f.readlines()
        filenames = [filename.strip() for filename in filenames]

    for filename in filenames:
        subfolder, file = filename.split('/')
        source_path = os.path.join(source_folder, filename)
        dest_path = os.path.join(destination_folder, subfolder, file)
        os.makedirs(os.path.join(destination_folder, subfolder), exist_ok=True)
        shutil.copy(source_path, dest_path)

def gather_files_in_folder(source_folder):
    files = []
    for root, _, filenames in os.walk(source_folder):
        for filename in filenames:
            relative_path = os.path.relpath(os.path.join(root, filename), source_folder)
            files.append(relative_path.replace('\\', '/'))  # Fix for Windows paths
    return files

def separate_subfolder(filename):
    if '/' in filename:
        return filename.split('/')
    else:
        return None, filename


def copy_background_noise(source_folder, split_data_path):
    background_noise_folder = os.path.join(source_folder, '_background_noise_')
    dest_noise_folder = os.path.join(split_data_path, '_background_noise_')
    if os.path.exists(background_noise_folder):
        shutil.copytree(background_noise_folder, dest_noise_folder, dirs_exist_ok=True)

def split_data(data_path,split_data_path):

    validation_list = os.path.join(data_path,'validation_list.txt')
    testing_list = os.path.join(data_path, 'testing_list.txt')
    source_folder = data_path
    train_folder = os.path.join(split_data_path, 'train')
    val_folder = os.path.join(split_data_path, 'validation')
    test_folder = os.path.join(split_data_path, 'test')

    # Create validation set
    create_directory_structure(validation_list, source_folder, val_folder)

    # Create testing set
    create_directory_structure(testing_list, source_folder, test_folder)

    # Create training set by excluding files in validation and testing sets
    all_files = gather_files_in_folder(source_folder)
    all_files = [file for file in all_files if not file.startswith('_background_noise_')]
    val_test_files = set()
    with open(validation_list, 'r') as f:
        val_test_files.update(f.read().splitlines())
    with open(testing_list, 'r') as f:
        val_test_files.update(f.read().splitlines())

    train_files = [file for file in all_files if file not in val_test_files]
    for filename in train_files:
        subfolder, file = separate_subfolder(filename)
        source_path = os.path.join(source_folder, filename)

        if subfolder:
            dest_path = os.path.join(train_folder, subfolder, file)
            os.makedirs(os.path.join(train_folder, subfolder), exist_ok=True)
            shutil.copy(source_path, dest_path)
        else:
            # Check if the file is a .wav file
            if file.lower().endswith('.wav'):
                dest_path = os.path.join(train_folder, file)
                shutil.copy(source_path, dest_path)


    copy_background_noise(data_path, split_data_path)

#def main():
#    DATASET_PATH = r'C:\Users\maren\OneDrive\HDA_Project\project_data'
#    DATASET_SPLIT_PATH = r'C:\Users\maren\OneDrive\HDA_Project\Test\project_data_split'
#    data_split(DATASET_PATH, DATASET_SPLIT_PATH)

#if __name__ == "__main__":
#    main()