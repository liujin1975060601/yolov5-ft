import os
import os.path as osp



def ucas_label_to_dota():
    p1 = '/home/LIESMARS/2019286190105/datasets/final-master/UCASALL/labels/train'
    p2 = '/home/LIESMARS/2019286190105/datasets/final-master/UCASALL/labelTxtAll'
    p3 = '/home/LIESMARS/2019286190105/datasets/final-master/UCASALL/labelTxt/train'

    classes = ['plane', 'car']
    
    files = list(filter(lambda x: x.endswith('.txt'), os.listdir(p1)))
    for file in files:
        print(file)
        with open(osp.join(p2, file), 'r') as f:
            new_lines = []
            for line in f:
                line = line.strip().split(' ')
                cls = int(line[0])
                new_line = line[1:9]
                new_line.append(classes[cls])
                new_lines.append(new_line)
        
        with open(osp.join(p3, file), 'w') as f:
            for line in new_lines:
                f.write(' '.join(line) + '\n')



if __name__ == '__main__':
    ucas_label_to_dota()