import yaml

basename = 'dgcnn_foveal_@.yaml'
master = 'dgcnn_foveal_door.yaml'
master_class = 'Door'

with open('cats.txt','r') as cats_file:
    for cat in cats_file.readlines():
        cat=cat.strip()
        cat = cat.replace('_small','')
        print(cat)
        with open(master) as stream:
            x=yaml.safe_load(stream)
            root_dir = x['DATASET']['ROOT_DIR']
            x['DATASET']['ROOT_DIR'] = root_dir.replace(master_class, cat)
            with open(basename.replace('@', cat.lower()),'w') as out_yaml:
                out_yaml.write(yaml.dump(x))

