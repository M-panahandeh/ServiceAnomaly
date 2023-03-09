
# This is a sample Python script.
import ast
import json

from networkx.algorithms import isomorphism

from DAG import DAG
############
from decimal import Decimal
import json
import os
import sklearn
import pandas as pd
from numpy import mean
from numpy import std
from numpy import cov
from matplotlib import pyplot
from matplotlib import pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
import matplotlib.pyplot as plt
#book info metrics
import ctypes  # An included library with Python install.
###########
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import tkinter as tk

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#for teastore:
from MetricLearningTeastore import MetricLearning
#from MetericsProfiling import MetericsProfiling (for other cases) for teastore, use the buttom line
from MetericsProfilingTeastore import MetericsProfiling
from TestTeastore import Test
from TraceCollection import TraceCollection



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()


    #lists inclduing all relationships forall edges
    ListOfLinearRelashion=[]
    ListOfNonlinearRelashion=[]
     #Collect traces
   ###############################
       # traces=TraceCollection()
       # traces.Trace2json()
    ###########################
    # Read traces json file to a json obj:
    with open('traces.json') as json_file:
       trace = json.load(json_file)
    #convert to json
    trace=json.loads(trace)
    # create a DAG:
    relations = trace['data']
    dag = DAG()
    dag.GrphGeneration(relations)
    dag.GraphVisualize()

    #Filtering and Grouping metrics

    metrics = MetericsProfiling()
   ###################################################

    # input should be labels or relations if the graph is not DAG
    #print(dag.edges)
    #use the print result as input of the following command
    #metrics.Mertics2json(dag.edges)
   ###################################################
    #Make a matrix of metrics
    #if the graph is DAG edges are label. So label can be easily get from print command in DAG-l29
    #dag.edges =[('productpage.default','details.default'),('productpage.default','reviews.default'),('reviews.default','rating.default')]
    #dag.edges=[('ts-travel-service.default', 'ts-train-service.default'), ('ts-travel-service.default', 'ts-seat-service.default'), ('ts-travel-service.default', 'ts-route-service.default'), ('ts-travel-service.default', 'ts-ticketinfo-service.default'), ('ts-travel2-service.default', 'ts-ticketinfo-service.default'), ('ts-travel2-service.default', 'ts-train-service.default'), ('ts-travel2-service.default', 'ts-route-service.default'), ('ts-ticketinfo-service.default', 'ts-basic-service.default'), ('ts-inside-payment-service.default', 'ts-payment-service.default'), ('ts-inside-payment-service.default', 'ts-order-service.default'), ('ts-inside-payment-service.default', 'ts-order-other-service.default'), ('ts-preserve-service.default', 'ts-user-service.default'), ('ts-preserve-service.default', 'ts-security-service.default'), ('ts-preserve-service.default', 'ts-contacts-service.default'), ('ts-preserve-service.default', 'ts-travel-service.default'), ('ts-preserve-service.default', 'ts-seat-service.default'), ('ts-preserve-service.default', 'ts-assurance-service.default'), ('ts-preserve-service.default', 'ts-food-service.default'), ('ts-preserve-service.default', 'ts-ticketinfo-service.default'), ('ts-preserve-service.default', 'ts-station-service.default'), ('ts-preserve-service.default', 'ts-consign-service.default'), ('ts-preserve-service.default', 'ts-order-service.default'), ('ts-seat-service.default', 'ts-order-other-service.default'), ('ts-seat-service.default', 'ts-travel-service.default'), ('ts-seat-service.default', 'ts-config-service.default'), ('ts-seat-service.default', 'ts-order-service.default'), ('ts-basic-service.default', 'ts-train-service.default'), ('ts-basic-service.default', 'ts-route-service.default'), ('ts-basic-service.default', 'ts-station-service.default'), ('ts-food-service.default', 'ts-station-service.default'), ('ts-food-service.default', 'ts-travel-service.default'), ('ts-ui-dashboard.default', 'ts-preserve-service.default'),('ts-ui-dashboard.default', 'ts-rebook-service.default'), ('ts-ui-dashboard.default', 'ts-contacts-service.default'), ('ts-ui-dashboard.default', 'ts-order-service.default'), ('ts-ui-dashboard.default', 'ts-order-other-service.default'), ('ts-ui-dashboard.default', 'ts-inside-payment-service.default'), ('ts-order-service.default', 'ts-station-service.default'), ('ts-order-other-service.default', 'ts-station-service.default'), ('ts-consign-service.default', 'ts-consign-price-service.default')]
    dag.edges=[('teastore-webui','teastore-auth.default.svc.cluster.local'),('teastore-webui','teastore-image.default.svc.cluster.local'),('teastore-webui','teastore-persistence.default.svc.cluster.local'),('teastore-webui','teastore-recommender'),('teastore-webui','teastore-registry.default.svc.cluster.local'),('teastore-image.default.svc.cluster.local','teastore-registry.default.svc.cluster.local'),('teastore-auth.default.svc.cluster.local','teastore-registry.default.svc.cluster.local'),('teastore-recommender','teastore-registry.default.svc.cluster.local'),('teastore-persistence.default.svc.cluster.local','teastore-registry.default.svc.cluster.local'),('teastore-image.default.svc.cluster.local','teastore-persistence.default.svc.cluster.local'),('teastore-recommender','teastore-persistence.default.svc.cluster.local'),('teastore-auth.default.svc.cluster.local','teastore-persistence.default.svc.cluster.local')]
    for edge in dag.edges:
        # remove jeager labels to make it calleable in prometheus
        destination = edge[1].split('.')[0]
        source=edge[0].split('.')[0]
        #Grouping
        #for teastore use this grouping because of file formats:
        GroupedMetrics = metrics.MetricGrouping(source,destination)
        #for othercase studies use the following function
        #GroupedMetrics= metrics.MetricGrouping(destination)

        # Metric learning
        ML=MetricLearning()
        # Linear:
        ML.LinearRelationship_Visualize(GroupedMetrics,source,destination)
        # get linear coefficient
        # 0.6 is the correlation bound
        print('Linear Regressions for '+edge[0].split('.')[0]+'-->'+destination+':\n')
        ML.get_top_abs_correlations(GroupedMetrics, 0.6)
        # add the dge to the lin-dictinary of relatinships
        #lin_dic: list of a dictionary inclduing all linear relshionships of an edge [[{one edge}],[{}]...]
        lin_dic = [{'source': source, 'destination': destination,'relations':ML.lin_ls}]
        # list of all relashonships for all edges
        ListOfLinearRelashion.append(lin_dic)


        #nonlinear relashionship:
        ML.NonLinearRelationship_Visualize(GroupedMetrics, source, destination)
        # get linear coefficient
        # 0.3 is the NMI bound
        print('Nonlinear Relationship for ' + source + '-->' + destination + ':\n')
        NMI_t= 0.6 #nmi threashold
        ML.get_non_linear(GroupedMetrics, NMI_t,0.6)
       # add the dge to the nonlin-dictinary of relatinships
        nonlin_dic = [{'source': source, 'destination': destination,'relations':ML.nonlin_ls}]
        # list of all relashonships for all edges
        ListOfNonlinearRelashion.append(nonlin_dic)

    #############print ang of relashionships per edges####################################
    lin_avg=metrics.Num_Relationships(ListOfLinearRelashion)
    ctypes.windll.user32.MessageBoxW(0, "The Average number of linear relationships is"+str(lin_avg),"",  1)
    #############print ang of relashionships per edges####################################
    lin_avg = metrics.Num_Relationships(ListOfNonlinearRelashion)
    ctypes.windll.user32.MessageBoxW(0, "The Average number of non-linear relationships is" + str(lin_avg), "", 1)
     ###############test#############################

    while (True):
            # test gets the trace
            # enter with []
            #[{'parent': 'productpage.default', 'child': 'reviews.default'},......]
            relation_test=input("Enter test relations: ")
            # eval() used to convert
            res = list(eval(relation_test))
            #create test-grapg
            test=Test()
            test.test_set_relation(res)
            #Make a copy of DAG
            g_main = nx.DiGraph()
            g_main.add_edges_from(dag.G.edges)

            #visualize test_trace
            test.test_graphGeneralization()
            test.Is_Isomorph(g_main)
            num_linear_violatiom=0
            num_nonlin_violation=0
            #Get metrics of injected fault for each edge and check against the found relations
            for edge in test.dag_test.edges:
                 # remove jeager labels to make it calleable in prometheus
                 destination_test = edge[1].split('.')[0]
                 source_test = edge[0].split('.')[0]
                 # Grouping for an edge in a dic
                 metric_test=test.test_metric_gathering(source_test,destination_test)
                 #check if metrics are fit in LR models
                 for edge_relation_ls in ListOfLinearRelashion:
                     #check linear
                     for dic in edge_relation_ls:
                         if dic["source"]==source_test and dic["destination"]==destination_test:
                             #get relations one by one
                             print('Total number of linear relationships for: '+source_test+' and '+destination_test+ ': '+
                                   str(len(dic["relations"])))
                             for rel in dic["relations"]:
                                 try:
                                     metric1= rel["metric1"]
                                     metric2= rel["metric2"]
                                     coef=float(rel["coef"][0])
                                     print("coef:= "+str(coef)) ### it should be removed
                                     intercept=float(rel["intercept"])
                                     error= float(rel["error"])
                                     y_test=metric_test[metric1]
                                     x_test=metric_test[metric2]
                                     print("y="+str(y_test)+" x="+str(x_test))
                                     #check fit
                                     #x_test = np.array([x_test]).reshape(-1, 1)
                                     #y_predict = rel['reg'].predict(x_test)
                                     y_predict=(coef*x_test)+intercept
                                     #y_predict=reg.predic(x)
                                     if abs(y_test-y_predict)>abs((error)):
                                         print(str(error)+" "+str(y_test-y_predict)+" "+str(y_predict))
                                         num_linear_violatiom=num_linear_violatiom+1
                                         print('\x1b[6;30;42m' + 'violation of linear relationship for '+source_test+' and '
                                               +destination_test+'\x1b[0m')
                                         print(rel)
                                         ##
                                         print('violation for :' + str(metric1) + 'dependent on ' + str(
                                                 metric2) + ' |Differece: ',
                                             str(y_predict - y_test), 'error: ' + str(error))
                                     else:
                                         print(
                                             'no violation for :' + str(metric1) + 'dependent on ' + str(
                                                 metric2) + ' |Differece: ',
                                             str(y_predict-y_test), 'error: ' + str(error))
                                 except:
                                     pass
                             print('number of violated linear relationships: '+str(num_linear_violatiom))
                             print('------------------------------------------------------------------------\n')

                 #check non-linear
                 for edge_relation_ls in ListOfNonlinearRelashion:
                         #check nonlinear
                         for dic in edge_relation_ls:
                             if dic["source"]==source_test and dic["destination"]==destination_test:
                                 print('number of total non-linear relationships for: ' +source_test+' and '+destination_test+': '
                                       + str(len(dic["relations"])))
                                 #get relations one by one
                                 for rel in dic["relations"]:
                                   try:
                                     metric1= rel["metric1"]
                                     metric2= rel["metric2"]
                                     svr=(rel["svr"])
                                     error= float(rel["error"])
                                     y_test=metric_test[metric1]
                                     x_test=metric_test[metric2]
                                     #check fit
                                     x_test=np.array([x_test]).reshape(-1,1)
                                     y_predict=svr.predict(x_test)
                                     if abs(y_test-y_predict)>abs(error):
                                         num_nonlin_violation=num_nonlin_violation+1
                                         print('\x1b[6;30;44m' + 'violation of Nonlinear relationship for'+source_test+' and '
                                               +destination_test+'\x1b[0m')
                                         print(rel)
                                         ###
                                         print(
                                             'violation for :' + str(metric1) + 'dependent on ' + str(
                                                 metric2) + ' |Differece: ',
                                             str(y_predict - y_test), 'error: ' + str(error))
                                     else:###
                                         print(
                                             'no violation for :' + str(metric1) + 'dependent on ' + str(
                                                 metric2) + ' |Differece: ',
                                             str(y_predict - y_test), 'error: ' + str(error))
                                   except:
                                       pass
                                 print('number of violated non-linear relationships: ' + str(num_nonlin_violation))
                                 print('-----------------------------------------------------------------------\n')

            ctypes.windll.user32.MessageBoxW(0, "The total number of violated linear relationships" +
                                                  str(num_linear_violatiom), "", 1)
            ctypes.windll.user32.MessageBoxW(0, "The Average number of non-linear relationships is" +
                                                  str(num_nonlin_violation), "", 1)











