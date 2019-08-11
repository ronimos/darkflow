# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:10:26 2019

@author: Ron Simenhois

This script  updates annotation xml files that were created with labelImg to be compatible with this fork of the darkflow implementation.
The labelImg VOC/Pascal format save the filename extension as xml. Darkflow use this 
field as the image file name when it parse and batch the data for training 
(see parse and _batch function in darkflow/darkflow/yolo/data.py and 
darkflow/darkflow/utils/pascal_voc_clean_xml.py).  
"""
import argparse
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

def _main(args):
    """ Change filename to the corespondet image filename"
    args: args.annotation_dir (str) - annotations' files directory
          args.im_file_type (str) - Images file extantion
    returns: None
    """
    path = args.annotation_dir
    im_ext = str(args.im_file_type).replace(".","")
    
    for file in tqdm([os.path.join(path, f) for f in os.listdir(path) if f.endswith(".xml")]):
        tree = ET.parse(file)
        filename = tree.find("filename")
        filename.text = str(filename.text).replace("xml", im_ext)
        tree.write(file)
        


parser = argparse.ArgumentParser()
parser.add_argument("-dir", dest="annotation_dir", help="Directory with the xml files")
parser.add_argument("-type", dest="im_file_type", help="Images file extantion")
    
if __name__=="__main__":
    
    args = parser.parse_args()
    _main(args)
