import json



with open('/home/hteam/Documents/han/paper/data/annotations.json') as json_file:
    annotation = json.load(json_file)
    test = {}
    train = {}
    test['images'] = annotation['images']
    train['images'] = annotation['images']
    test['type'] = annotation['type']
    train['type'] = annotation['type']
    test['annotations'] = []
    train['annotations'] = []

    count = 0
    for a in annotation['annotations']:
        if count % 250 == 0:
            #test
            test['annotations'].append(a)
        else:
            #train
            train['annotations'].append(a)

        count += 1

    test['categories'] = annotation['categories']
    train['categories'] = annotation['categories']

    with open('test.json', 'w') as outfile:
        json.dump(test, outfile)
    with open('train.json', 'w') as outfile:
        json.dump(train, outfile)
    