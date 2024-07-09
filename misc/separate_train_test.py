import os
import shutil
import random

def move_10_percent_files(source_folder, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get a list of all files in the source folder
    files = os.listdir(source_folder)

    # Calculate how many files constitute 10% of the total
    num_files_to_move = int(len(files) * 0.1)

    # Randomly select 10% of the files
    files_to_move = random.sample(files, num_files_to_move)

    for file in files_to_move:
        # Construct the full path of the source and destination files
        source_file = os.path.join(source_folder, file)
        destination_file = os.path.join(destination_folder, file)

        # Move the file to the destination folder
        shutil.move(source_file, destination_file)
        print(f"Moved {file} to {destination_folder}")

# Example usage:
source_folder = '../cropped_data_train'
destination_folder = '../cropped_data_test'
move_10_percent_files(source_folder, destination_folder)