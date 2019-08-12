"""
This script used k-mean over Intersection over Union metric. This script is insiered 
by Lars' Blog (https://lars76.github.io/object-detection/k-means-anchor-boxes/  )         
"""
import random
import argparse
import numpy as np
import xml.etree.ElementTree as ET
import os
import json

def parse_annotation(ann_dir, labels):
    """ parse the xml annotations file to list of images properties and bonding boxes
    args: ann_dir (str) annotations' directory
    labels : list(str) image object lables
    return all_insts - list of dicts: [{img{"filename":str, "width": int, "heigth": int, 
                                            "obj": [{"name": str (label), "xmin":float,
                                            "ynim:float, "xmax":float,"ymax":float}]}}...]
           seen_labels - dict: counter of how many times each label appear in the dataset
"""
    all_insts = []
    seen_labels = {}    
    for ann in sorted([ann for ann in os.listdir(ann_dir) if ann.endswith(".xml")]):
        img = {"object":[]}

        try:
            tree = ET.parse(os.path.join(ann_dir, ann))
        except Exception as e:
            print(e)
            print("Ignore this bad annotation: " + ann_dir + ann)
            continue
        img["filename"] = tree.find("filename").text
        elem = tree.find("size")
        img["width"] = int(elem.find("width").text)
        img["height"] = int(elem.find("height").text)
        
        for _obj in tree.iter("object"): 
            obj = {}
            obj["name"] = _obj.find("name").text
            if obj["name"] in seen_labels:
                seen_labels[obj["name"]] += 1
            else:
                seen_labels[obj["name"]] = 1
            if len(labels) > 0 and obj["name"] not in labels:
                break
            img["object"] += [obj]
            for bbox in _obj.iter("bndbox"):
                obj["xmin"] = float(bbox.find("xmin").text)
                obj["ymin"] = float(bbox.find("ymin").text)
                obj["xmax"] = float(bbox.find("xmax").text)
                obj["ymax"] = float(bbox.find("ymax").text)
                assert obj["xmin"] < obj["xmax"] and obj["ymin"] < obj["ymax"], \
                "Annot error at {0}".format(img["filename"])
        if len(img["object"]) > 0:
            all_insts += [img]

    return all_insts, seen_labels

def IOU(ann, centroids):
    """Calculates the Intersection over Union (IoU) between a ann and k clusters' centroids."""
    
    w_intersection = np.minimum(centroids[:,0], ann[0])
    h_intersection = np.minimum(centroids[:,1], ann[1])
    assert np.count_nonzero(w_intersection==0)==0 or np.count_nonzero(h_intersection==0)==0, \
           "Box has no area"
    
    intersections = w_intersection * h_intersection
    ann_area = ann[0] * ann[1]
    centroids_areas = centroids[:,0] * centroids[:,1]
    union = ann_area + centroids_areas - intersections    
    iou_ = intersections / union

    return iou_

def avg_IOU(anns, centroids):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param anns: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(IOU(ann, centroids)) for ann in anns])
    
def print_anchors(centroids):

    anchors = np.round(centroids, decimals=5)
    sorted_indices = np.argsort(anchors[:, 0])
    anchors = anchors[sorted_indices]
    out_string = ", ".join(anchors.astype(str).reshape(-1).tolist())
    print(out_string)

def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances) # distances.shape = (ann_num, anchor_num)

        print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

        #assign samples to centroids
        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all() :
            return centroids

        #calculate new centroids
        centroid_sums=np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]]+=ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()

def _main(args):
    config_path = args.conf
    num_anchors = int(args.anchors)

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    n_grid = config["model"]["n_grid"]
    ann_dir=config["train"]["train_annot_folder"]
    
    with open(config["model"]["labels"],"r") as lb:
        labels = lb.read().split(",")
    train_imgs, train_labels = parse_annotation(ann_dir=ann_dir, labels=labels)

    # run k_mean to find the anchors
    
    annotation_dims = []
    for image in train_imgs:
        for obj in image["object"]:
            #For yolo2 the relative width and heigth should be relative to the grid size
            relative_w = (float(obj["xmax"]) - float(obj["xmin"]))/image["width"]
            relative_h = (float(obj["ymax"]) - float(obj["ymin"]))/image["height"]
            annotation_dims.append(tuple(map(float, (relative_w,relative_h))))

    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, num_anchors)*[n_grid, n_grid]

    # write anchors to file
    print("\naverage IOU for", num_anchors, "anchors:", "%0.2f" % avg_IOU(annotation_dims, centroids))
    print_anchors(centroids)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", 
                        "--conf", 
                        help="address of config parameters json file", 
                        default="gen_anchors_config.json")
    parser.add_argument("-a",
                        "--anchors",
                        help="number od anchors",
                        default=5)
    args = parser.parse_args()
    _main(args)
