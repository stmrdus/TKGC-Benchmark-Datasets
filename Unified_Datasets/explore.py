import os
import pandas as pd
import numpy as np


def load_quadruples(file_path: str = "./GDELT/train.txt"):
    if not os.path.exists(file_path):
            raise FileNotFoundError
    
    with open(file_path, 'r') as f:
        # list quadruple
        quadrupleList = []
        
        times = set()
        entities = set()
        relations = set()
        
        for line in f:
            # reading a input line
            line_split = line.split()
            
            # take subject, object, relation, timestamp
            # subject = int(line_split[0])
            # object = int(line_split[2])
            # relation = int(line_split[1])
            # timestamp = int(line_split[3])
            
            subject = (line_split[0])
            object = (line_split[2])
            relation = (line_split[1])
            timestamp = (line_split[3])
            
            quadrupleList.append([subject, relation, object, timestamp])
            times.add(timestamp)
            
            entities.add(subject)
            entities.add(object)
            relations.add(relation)
            
        times = list(times)
        times.sort()
        
    return list(quadrupleList), list(entities), list(relations), times


def data_loader(data_path: str = "./GDELT", type_loader: str = "train"):
    if type_loader == "train":
        train_dir = data_path + "/train.txt"
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError
        
        quadrupleList, entitiesSet, relationSet, timesSet = load_quadruples(train_dir)
        
        print(f"Number of quadruples in train: {len(quadrupleList)}")
        print(f"Number of entities in train: {len(entitiesSet)}")
        print(f"Number of relations in train: {len(relationSet)}")
        
    if type_loader == "valid":
        valid_dir = data_path + "/valid.txt"
        
        if not os.path.exists(valid_dir):
            raise FileNotFoundError
        
        quadrupleList, entitiesSet, relationSet, timesSet = load_quadruples(valid_dir)
        
        print(f"Number of quadruples in valid: {len(quadrupleList)}")
        print(f"Number of entities in valid: {len(entitiesSet)}")
        print(f"Number of relations in valid: {len(relationSet)}")
        
    if type_loader == "test":
        test_dir = data_path + "/test.txt"
        
        if not os.path.exists(test_dir):
            raise FileNotFoundError
        
        quadrupleList, entitiesSet, relationSet, timesSet = load_quadruples(test_dir)
        
        print(f"Number of quadruples in test: {len(quadrupleList)}")
        print(f"Number of entities in test: {len(entitiesSet)}")
        print(f"Number of relations in test: {len(relationSet)}")
        
    if type_loader == "all":
        
        quadrupleList = [] 
        entitiesSet = []
        relationSet = []
        timesSet = []  
        
        train_dir = data_path + "/train.txt"
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError
        
        trainQuadrupleList, trainEntitiesSet, trainRelationSet, trainTimesSet = load_quadruples(train_dir)
        
        print(f"Number of quadruples in train: {len(trainQuadrupleList)}")
        print(f"Number of entities in train: {len(set(trainEntitiesSet))}")
        print(f"Number of relations in train: {len(set( trainRelationSet))}")
        print(f"Number of timestamp in train: {len(trainTimesSet)}")
        print(f"------------------------------------------------------------")
        
        quadrupleList += trainQuadrupleList
        entitiesSet = entitiesSet + trainEntitiesSet
        relationSet = relationSet + trainRelationSet
        timesSet = timesSet + trainTimesSet
        
        del trainQuadrupleList
        del trainEntitiesSet
        del trainRelationSet
        del trainTimesSet        
        
        valid_dir = data_path + "/valid.txt"
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError
        
        validQuadrupleList, validEntitiesSet, validRelationSet, validTimesSet = load_quadruples(valid_dir)
        
        print(f"Number of quadruples in valid: {len(validQuadrupleList)}")
        print(f"Number of entities in valid: {len(set(validEntitiesSet))}")
        print(f"Number of relations in valid: {len(set(validRelationSet))}")
        print(f"Number of timestamp in valid: {len(validTimesSet)}")
        print(f"------------------------------------------------------------")
        
        quadrupleList += validQuadrupleList
        entitiesSet = entitiesSet + validEntitiesSet
        relationSet = relationSet + validRelationSet
        timesSet = timesSet + validTimesSet
        
        del validQuadrupleList
        del validEntitiesSet
        del validRelationSet
        del validTimesSet
        
        test_dir = data_path + "/test.txt"
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError
        
        testQuadrupleList, testEntitiesSet, testRelationSet, testTimesSet = load_quadruples(test_dir)
        
        print(f"Number of quadruples in test: {len(quadrupleList)}")
        print(f"Number of entities in test: {len(testEntitiesSet)}")
        print(f"Number of relations in test: {len(testRelationSet)}")
        print(f"Number of timestamp in test: {len(testTimesSet)}")
        print(f"------------------------------------------------------------")
        
        quadrupleList += testQuadrupleList
        entitiesSet = entitiesSet + testEntitiesSet
        relationSet = relationSet + testRelationSet
        timesSet = timesSet + testTimesSet
        
        del testQuadrupleList
        del testEntitiesSet
        del testRelationSet
        del testTimesSet
        
        entitiesSet = set(entitiesSet)
        relationSet = set(relationSet)
        
        print(f"Number of quadruples: {len(quadrupleList)}")
        print(f"Number of entities: {len(entitiesSet)}")
        print(f"Number of relations: {len(relationSet)}")
        print(f"Number of timestamp: {len(timesSet)}")
        print(f"------------------------------------------------------------")
        
        return quadrupleList, entitiesSet, relationSet, timesSet
        
        
        
data_loader(data_path="../Text_Datasets/icews18", type_loader="all")      
        
        