project_dir = '/Users/sondaehyeon/Desktop/OCR'

data = {
    'img_path': project_dir + '/data/images',
    'label_path': project_dir + '/data/labels.txt',
}

preprocessing = {
    'char2int_path': project_dir + '/preprocessing/char2int.pkl',
    'int2char_path': project_dir + '/preprocessing/int2char.pkl',
    'csv_path': project_dir + '/preprocessing/data_file.csv',
}

training = {
    'epochs': 100,
    'batch_size': 32,
    'output_path': project_dir + '/output/model.pth'
}

