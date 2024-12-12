import os
import csv
def write_folder_names_to_csv(folder_path, csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Folder Name'])

        for folder_name in os.listdir(folder_path):
            folder_dir = os.path.join(folder_path, folder_name)
            if os.path.isdir(folder_dir):
                csv_writer.writerow([folder_name])


if __name__ == "__main__":
    # 请确保替换为实际的文件夹路径和CSV文件路径
    folder_path = "../datasets/huakenew_70/diease_folder"
    csv_file_path = "../datasets/huakenew_70/diease_folder.csv"
    write_folder_names_to_csv(folder_path, csv_file_path)
