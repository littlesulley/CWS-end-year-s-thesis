import os 


def merge_files(BASE_DIR, files, save_file_path):
    for file in files:
        with open(os.path.join(BASE_DIR, file), 'r', encoding='utf-8') as fin:
            with open(save_file_path, 'a', encoding='utf-8') as fout:
                lines = fin.readlines()
                lines = [line for line in lines if not line.startswith('<')]
                fout.writelines(lines)

test_data_files = ['chtb_000' + str(i) + '.seg' for i in range(1, 10)] + ['chtb_00' + str(i) + '.seg' for i in range(10, 41)] + \
                    ['chtb_0' + str(i) + '.seg' for i in range(901, 932)] + ['chtb_1018.seg', 'chtb_1020.seg', 'chtb_1036.seg'] + \
                    ['chtb_1044.seg', 'chtb_1060.seg', 'chtb_1061.seg', 'chtb_1072.seg', 'chtb_1118.seg', 'chtb_1119.seg', 'chtb_1132.seg'] + \
                    ['chtb_1141.seg','chtb_1142.seg', 'chtb_1148.seg'] + ['chtb_' + str(i) + '.seg' for i in range(2165, 2181)] + \
                    ['chtb_' + str(i) + '.seg' for i in range(2296, 2311)] + ['chtb_' + str(i) + '.seg' for i in range(2570, 2603)] + \
                    ['chtb_' + str(i) + '.seg' for i in range(2800, 2820)] + ['chtb_' + str(i) + '.seg' for i in range(3110, 3146)]


valid_data_files = ['chtb_00' + str(i) + '.seg' for i in range(41, 80)] + ['chtb_' + str(i) + '.seg' for i in range(1120, 1130)] + \
                    ['chtb_' + str(i) + '.seg' for i in range(2140, 2160)] + ['chtb_' + str(i) + '.seg' for i in range(2280, 2295)] + \
                    ['chtb_' + str(i) + '.seg' for i in range(2550, 2570)] + ['chtb_' + str(i) + '.seg' for i in range(2775, 2800)] + \
                    ['chtb_' + str(i) + '.seg' for i in range(3080, 3110)]


train_data_files = ['chtb_00' + str(i) + '.seg' for i in range(81, 100)] + ['chtb_0' + str(i) + '.seg' for i in range(100, 326)] + ['chtb_0' + str(i) + '.seg' for i in range(400, 455)] + \
                    ['chtb_0' + str(i) + '.seg' for i in range(600, 886)] + ['chtb_0900.seg'] + \
                    ['chtb_0' + str(i) + '.seg' for i in range(500, 555)] + ['chtb_0' + str(i) + '.seg' for i in range(590, 597)] + \
                    ['chtb_' + str(i) + '.seg' for i in range(1001, 1018)] + ['chtb_1019.seg'] + \
                    ['chtb_' + str(i) + '.seg' for i in range(1021, 1036)] + ['chtb_' + str(i) + '.seg' for i in range(1037, 1044)] + \
                    ['chtb_' + str(i) + '.seg' for i in range(1045, 1060)] + ['chtb_' + str(i) + '.seg' for i in range(1062, 1072)] + \
                    ['chtb_' + str(i) + '.seg' for i in range(1073, 1079)] + ['chtb_' + str(i) + '.seg' for i in range(1100, 1118)] + \
                    ['chtb_1130.seg', 'chtb_1131.seg'] + ['chtb_' + str(i) + '.seg' for i in range(1133, 1141)] + \
                    ['chtb_' + str(i) + '.seg' for i in range(1143, 1148)] + ['chtb_' + str(i) + '.seg' for i in range(1149, 1152)] + \
                    ['chtb_' + str(i) + '.seg' for i in range(2000, 2140)] + ['chtb_' + str(i) + '.seg' for i in range(2160, 2165)] + \
                    ['chtb_' + str(i) + '.seg' for i in range(2181, 2280)] + ['chtb_' + str(i) + '.seg' for i in range(2311, 2550)] + \
                    ['chtb_' + str(i) + '.seg' for i in range(2603, 2775)] + ['chtb_' + str(i) + '.seg' for i in range(2820, 3080)]

BASE_DIR = '/home/sulley/桌面/datasets/ctb6/data/segmented'
merge_files(BASE_DIR, train_data_files, './datasets/ctb_train')
merge_files(BASE_DIR, valid_data_files, './datasets/ctb_dev')
merge_files(BASE_DIR, test_data_files, './datasets/ctb_test')