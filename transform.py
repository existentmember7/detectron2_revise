import os
import yaml
import json

def category(name):
    c = ["person", "wheelchair", "push_wheelchair", "crutches", "walking_frame"]
    ans = c.index(name) + 1
    return ans

re_id = []

for filename in os.listdir(os.getcwd()+'/Images_RGB'):
    re_id.append(filename.split('_')[1].split('.png')[0])

annotation = dict()
annotation["image"] = []
annotation["type"] = "instances"
annotation["annotations"] = []
annotation["categories"] = [
{"supercategory": "none", "id": 1, "name": "person"},
{"supercategory": "none", "id": 2, "name": "wheelchair"},
{"supercategory": "none", "id": 3, "name": "push_wheelchair"},
{"supercategory": "none", "id": 4, "name": "crutches"},
{"supercategory": "none", "id": 5, "name": "walking_frame"},
]

count = 0
iteration = 0
for filename in os.listdir(os.getcwd()+'/Annotations_RGB'):
    
    with open(r'Annotations_RGB/'+filename) as file:
        list = yaml.load(file, Loader=yaml.FullLoader)
        iteration += 1
        print(iteration)
        
        # image
        id = list["annotation"]["filename"].split('_')[1].split('.png')[0]
        annotation["image"].append({
        "file_name": list["annotation"]["filename"],
        "height": int(list["annotation"]["size"]["height"]),
        "width": int(list["annotation"]["size"]["width"]),
        "id": re_id.index(id) })

        try:
            test = list["annotation"]["object"]
        except:
            continue

        for object in list["annotation"]["object"]:
            box_width = int(object["bndbox"]["xmax"]) - int(object["bndbox"]["xmin"])
            box_height = int(object["bndbox"]["ymax"]) - int(object["bndbox"]["ymin"])
            x = int(object["bndbox"]["xmin"])
            y = int(object["bndbox"]["ymin"])
            
            # anntation
            annotation["annotations"].append({
            "area": box_width * box_height,
            "iscrowd": 0,
            "image_id": re_id.index(id),
            "bbox": [x, y, box_width, box_height],
            "category_id": category(object["name"]),
            "id": count,
            "ignore": 0,
            "segmentation": []})
            
            count += 1
            
#            print(object["name"])

with open('annotations.json', 'w') as fp:
    json.dump(annotation, fp)



