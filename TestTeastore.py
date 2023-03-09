from networkx.algorithms import isomorphism
import tkinter as tk
from tkinter import filedialog
from tkinter import filedialog
import pandas as pd
import json
from DAG import DAG


class Test:
    dag_test = DAG()
    # input the relation_test
    relations_test = []
    def test_set_relation(self,relation_test):
        self.relations_test = relation_test
        # [{'parent': 'productpage.default', 'child': 'reviews.default'},
        #  {'parent': 'reviews.default', 'child': 'ratings.default'},
        #  {'parent': 'productpage.default', 'child': 'details.default'}]
    #visualize graph
    def test_graphGeneralization(self):
      self.dag_test.GrphGeneration(self.relations_test)
      self.dag_test.GraphVisualize()

    # is test graph isomorphic of the original one
    def Is_Isomorph(self,g):
        GM = isomorphism.GraphMatcher(g, self.dag_test.G)
        print(GM.subgraph_is_isomorphic())

    #gather metrics of test trace
    def test_metric_gathering(self,source,destination):

            #Read metrics from files and put in a matrix
            #get the path :
            root = tk.Tk()
            root.withdraw()
            #file_path = filedialog.askopenfilename()
            root.attributes('-topmost', True)  # Opened windows will be active. above all windows despite of selection.
            file_path = filedialog.askdirectory()  # Returns opened path as str
            #print(file_path)

            #request_duratoin
            with open(file_path+'/'+source+'2'+destination+'_request_duratoin.json') as json_file:
                request_duration = json.load(json_file)

            # convert to json
            request_duration = json.loads(request_duration)
            request_duration_result = request_duration['data']['result']
            request_duration_values = []
            for ls in request_duration_result:
                request_duration_values.extend(ls['values'])
            try:
                request_duration_values=float(request_duration_values[0][1])
            except:
                pass
            #####################same for the rest ofmetrics##################
            #request_byte
            with open(file_path + '/' +source+'2'+ destination + '_request_byte.json') as json_file:
                request_byte = json.load(json_file)

                # convert to json
            request_byte = json.loads(request_byte)
            request_byte_result = request_byte['data']['result']
            request_byte_values = []
            for ls in request_byte_result:
                request_byte_values.extend(ls['values'])
            try:
                  request_byte_values = float(request_byte_values[0][1])
            except:
                pass

            #################################################
            #response_byte
            with open(file_path + '/' +source+'2'+ destination + '_response_byte.json') as json_file:
                response_byte = json.load(json_file)

                # convert to json
            response_byte = json.loads(response_byte)
            response_byte_result = response_byte['data']['result']
            response_byte_values = []
            for ls in response_byte_result:
                response_byte_values.extend(ls['values'])
            try:
                response_byte_values = float(response_byte_values[0][1])
            except:
                pass
            ###############################################
            # Queue Size
            with open(file_path + '/' +source+'2'+ destination + '_queue_size.json') as json_file:
                queue_size = json.load(json_file)

            # convert to json
            queue_size = json.loads(queue_size)
            queue_size_result = queue_size['data']['result']
            queue_size_values = []
            for ls in queue_size_result:
                queue_size_values.extend(ls['values'])
            try:
                queue_size_values = float(queue_size_values[0][1])
            except:
                pass

            ###############################################
            # Latency
            with open(file_path + '/' +source+'2'+ destination + '_latency.json') as json_file:
                latency = json.load(json_file)

            # convert to json
            latency = json.loads(latency)
            latency_result = latency['data']['result']
            latency_values = []
            for ls in latency_result:
                latency_values.extend(ls['values'])
            try:
             latency_values = float(latency_values[0][1])
            except:
                pass
            ###############################################
            # Throughput
            with open(file_path + '/' +source+'2'+ destination + '_throughput.json') as json_file:
                throughput = json.load(json_file)

            # convert to json
            throughput = json.loads(throughput)
            throughput_result = throughput['data']['result']
            throughput_values = []
            for ls in throughput_result:
                throughput_values.extend(ls['values'])
            try:
                throughput_values = float(throughput_values[0][1])
            except:
                pass
            ################################################
            #Put all metrics in a dictionary

           # metrics= pd.merge(request_duration_values, request_byte_values, response_byte_values,queue_size_values,latency_values,throughput_values,on="movie_title")
            df_metric_test = {
                'request_duration': request_duration_values,
                'request_byte': request_byte_values,
                'response_byte': response_byte_values,
                'queue_size': queue_size_values,
                'latency': latency_values,
                #'throughput': throughput_values,
            }
            #normalized
            return df_metric_test