
import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime

import matplotlib.pyplot as plt
from functools import reduce

# class definition
class MetericsProfiling:
    data = []


    # filtering and grouping
    def Mertics2json(self, edges):
        for edge in edges:
            # remove jeager labels to make it calleable in prometheus
            destination = edge[1].split('.')[0]
            source = edge[0].split('.')[0]
            # print(destination,source)

            # filtering and grouping metrics

            # request duration_sum
            url = 'http://localhost:9090/api/v1/query?query=(istio_request_duration_milliseconds_sum{destination_service_name="' + destination + '",reporter="source",source_app="' + source + '",response_code="200"}[1d])'
            resp = requests.get(url)

            self.data = resp.content
            # print(self.data)
            self.data = self.data.decode('utf-8')  # string
            # write to Josn
            with open('./bookinfo/' + destination + '_request_duratoin.json', 'w') as outfile:
                json.dump(self.data, outfile)
                ##########################################################
            # request byte_sum
            url = 'http://localhost:9090/api/v1/query?query=(istio_request_bytes_sum{destination_service_name="' + destination + '", reporter="source",source_app="' + source + '", response_code="200"}[1d])'
            resp = requests.get(url)

            self.data = resp.content
            # print(self.data)
            self.data = self.data.decode('utf-8')  # string
            # write to Josn
            with open('./bookinfo/' + destination + '_request_byte.json', 'w') as outfile:
                json.dump(self.data, outfile)

            ####################################################################
            # latency
            url = 'http://localhost:9090/api/v1/query?query=(istio_agent_outgoing_latency{app="' + destination + '"}[1d])'
            resp = requests.get(url)

            self.data = resp.content
            # print(self.data)
            self.data = self.data.decode('utf-8')  # string
            # write to Josn
            with open('./bookinfo/' + destination + '_latency.json', 'w') as outfile:
                json.dump(self.data, outfile)
            ########################################################################3
            # response byte_sum

            url = 'http://localhost:9090/api/v1/query?query=(istio_response_bytes_sum{destination_service_name="' + destination + '",source_app="' + source + '",reporter="source",response_code="200"}[1d])'
            resp = requests.get(url)

            self.data = resp.content
            # print(self.data)
            self.data = self.data.decode('utf-8')  # string
            # write to Josn
            with open('./bookinfo/' + destination + '_response_byte.json', 'w') as outfile:
                json.dump(self.data, outfile)
            #################################################################################
            # Queue size_sum

            url = 'http://localhost:9090/api/v1/query?query=(istio_agent_pilot_proxy_queue_time_sum{app="' + destination + '"}[1d])'
            resp = requests.get(url)

            self.data = resp.content
            # print(self.data)
            self.data = self.data.decode('utf-8')  # string
            # write to Josn
            with open('./bookinfo/' + destination + '_queue_size.json', 'w') as outfile:
                json.dump(self.data, outfile)
            #################################################################################
            # RequestThroupout

            url = 'http://localhost:9090/api/v1/query?query=((rate(istio_requests_total{destination_app="' + destination + '",reporter="source", source_app="' + source + '", response_code="200"}[1m]))[1d:1m])'
            resp = requests.get(url)

            self.data = resp.content
            # print(self.data)
            self.data = self.data.decode('utf-8')  # string
            # write to Josn
            with open('./bookinfo/' + destination + '_throuput_size.json', 'w') as outfile:
                json.dump(self.data, outfile)
            #################################################################################
            # end for:saved all metrics for each edge

    #make matrices of metrics for each edge
    # def MetricGrouping(self,destination):
    #
    #         #Read metrics from files and put in a matrix
    #         #request_duratoin
    #         with open(destination+'_request_duratoin.json') as json_file:
    #             request_duration = json.load(json_file)
    #
    #         # convert to json
    #         request_duration = json.loads(request_duration)
    #         request_duration_result = request_duration['data']['result']
    #         request_duration_values = []
    #         for ls in request_duration_result:
    #             request_duration_values.extend(ls['values'])
    #         #initiate a matrix for the metric
    #         request_duration_metrics =[]
    #         for ls in request_duration_values:
    #             request_duration_metrics.append(float(ls[1]))
    #
    #         #plt.plot(request_duration_metrics)
    #         #plt.show()
    #         #####################same for the rest ofmetrics##################
    #         #request_byte
    #         with open(destination+'_request_byte.json') as json_file:
    #             request_byte = json.load(json_file)
    #         # convert to json
    #         request_byte = json.loads(request_byte)
    #         request_byte_result = request_byte['data']['result']
    #         request_byte_values = []
    #         for ls in request_byte_result:
    #             request_byte_values.extend(ls['values'])
    #         ##initiate a matrix for the metric
    #         request_byte_metrics =[]
    #         for ls in request_byte_values:
    #             request_byte_metrics.append(float(ls[1]))
    #         #################################################
    #         #response_byte
    #         with open(destination+'_response_byte.json') as json_file:
    #             response_byte = json.load(json_file)
    #         # convert to json
    #         response_byte = json.loads(response_byte)
    #         response_byte_result = response_byte['data']['result']
    #         response_byte_values = []
    #         for seri in response_byte_result:
    #             response_byte_values.extend(seri['values'])
    #         #initiate a matrix for the metric
    #         response_byte_metrics =[]
    #         for ls in response_byte_values:
    #             response_byte_metrics.append(float(ls[1]))
    #
    #         ###############################################
    #         # Latency
    #         with open(destination+'_latency.json') as json_file:
    #             latency = json.load(json_file)
    #         # convert to json
    #         latency = json.loads(latency)
    #         latency_result = latency['data']['result']
    #         latency_values = []
    #         for ls in  latency_result:
    #             latency_values.extend(ls['values'])
    #         ##initiate a matrix for the metric
    #         latency_metrics =[]
    #         for ls in latency_values:
    #             latency_metrics.append(float(ls[1]))
    #
    #         ###############################################
    #         # Queue Size
    #         with open(destination + '_queue_size.json') as json_file:
    #             queue_size = json.load(json_file)
    #             # convert to json
    #         queue_size = json.loads(queue_size)
    #         queue_size_result = queue_size['data']['result']
    #         queue_size_values = []
    #         for ls in queue_size_result:
    #             queue_size_values.extend(ls['values'])
    #         #initiate a matrix for the metric
    #         queue_size_metrics =[]
    #         for ls in queue_size_values:
    #             queue_size_metrics.append(float(ls[1]))
    #
    #         ###############################################
    #         # Throughput
    #         with open(destination + '_throughput.json') as json_file:
    #             throughput = json.load(json_file)
    #             # convert to json
    #         throughput = json.loads(throughput)
    #         throughput_result = throughput['data']['result']
    #         throughput_values = []
    #         for ls in throughput_result:
    #             throughput_values.extend(ls['values'])
    #         #initiate a matrix for the metric
    #         throughput_metrics =[]
    #         for ls in throughput_values:
    #             throughput_metrics.append(float(ls[1]))
    #
    #         ################################################
    #         #Put all metrics in a dictionary
    #         metrics = {'request_duration': request_duration_metrics,
    #                           'request_byte': request_byte_metrics,
    #                           'response_byte':response_byte_metrics,
    #                           'queue_size': queue_size_metrics,
    #                           'latency': latency_metrics,
    #                           'throughput':throughput_metrics}
    #         #print(len(metrics[request_duration]))
    #         return metrics


    def MetricSampling(self,matrix):
        #print(matrix)
        #find the min lenght of metrics in the matrix
        min_length = min(map(len, matrix.values()))
        #print(min_length)
        SamplingRange=min_length
        # Sampling
        ####sampling since lenght of metrics is not the same for all
        matrix ['request_duration']=matrix ['request_duration'][0:SamplingRange]
        matrix['request_byte'] = matrix['request_byte'][0:SamplingRange]
        matrix['response_byte'] = matrix['response_byte'][0:SamplingRange]
        matrix['queue_size'] = matrix['queue_size'][0:SamplingRange]
        matrix['latency'] = matrix['latency'][0:SamplingRange]
        matrix['throughput'] = matrix['throughput'][0:SamplingRange]
        #print(matrix)
        return matrix

    # make matrices of metrics for each edge
    def MetricGrouping(self,destination):
            #Read metrics from files and put in a matrix
            #request_duratoin
            with open(destination+'_request_duratoin.json') as json_file:
                request_duration = json.load(json_file)

            # convert to json
            request_duration = json.loads(request_duration)
            request_duration_result = request_duration['data']['result']
            request_duration_values = []
            for ls in request_duration_result:
                request_duration_values.extend(ls['values'])

            request_duration_values=pd.DataFrame.from_records(request_duration_values,columns=['time','value'])
            request_duration_values['value'] = request_duration_values['value'].astype(float)
            # groupby time and measure mean: it happens when data is for all service ves=rsions
            request_duration_values = request_duration_values.groupby('time').mean().reset_index()
            #####################same for the rest ofmetrics##################
            #request_byte
            with open(destination+'_request_byte.json') as json_file:
                request_byte = json.load(json_file)
            # convert to json
            request_byte = json.loads(request_byte)
            request_byte_result = request_byte['data']['result']
            request_byte_values = []
            for ls in request_byte_result:
                request_byte_values.extend(ls['values'])
            request_byte_values = pd.DataFrame.from_records(request_byte_values, columns=['time', 'value'])
            request_byte_values['value'] = request_byte_values['value'].astype(float)
            # groupby time and measure mean: it happens when data is for all service ves=rsions
            request_byte_values = request_byte_values.groupby('time').mean().reset_index()
            #################################################
            #response_byte
            with open(destination+'_response_byte.json') as json_file:
                response_byte = json.load(json_file)
            # convert to json
            response_byte = json.loads(response_byte)
            response_byte_result = response_byte['data']['result']
            response_byte_values = []
            for seri in response_byte_result:
                response_byte_values.extend(seri['values'])
            response_byte_values = pd.DataFrame.from_records(response_byte_values, columns=['time', 'value'])
            response_byte_values['value'] = response_byte_values['value'].astype(float)
            # groupby time and measure mean: it happens when data is for all service ves=rsions
            response_byte_values = response_byte_values.groupby('time').mean().reset_index()

            ###############################################
            # Queue Size
            with open(destination + '_queue_size.json') as json_file:
                queue_size = json.load(json_file)
                # convert to json
            queue_size = json.loads(queue_size)
            queue_size_result = queue_size['data']['result']
            queue_size_values = []
            for ls in queue_size_result:
                queue_size_values.extend(ls['values'])
            queue_size_values = pd.DataFrame.from_records(queue_size_values, columns=['time', 'value'])
            queue_size_values['value'] = queue_size_values['value'].astype(float)
            # groupby time and measure mean: it happens when data is for all service ves=rsions
            queue_size_values = queue_size_values.groupby('time').mean().reset_index()

            ###############################################
            # Latency
            with open(destination + '_latency.json') as json_file:
                latency = json.load(json_file)
            # convert to json
            latency = json.loads(latency)
            latency_result = latency['data']['result']
            latency_values = []
            for ls in latency_result:
                latency_values.extend(ls['values'])
            latency_values = pd.DataFrame.from_records(latency_values, columns=['time', 'value'])
            latency_values['value'] = latency_values['value'].astype(float)
            # groupby time and measure mean: it happens when data is for all service versions
            latency_values = latency_values.groupby('time').mean().reset_index()
            #print( latency_values['value'])
            ###############################################
            # Throughput
            with open(destination + '_throughput.json') as json_file:
                throughput = json.load(json_file)
                # convert to json
            throughput = json.loads(throughput)
            throughput_result = throughput['data']['result']
            throughput_values = []
            for ls in throughput_result:
                throughput_values.extend(ls['values'])
            throughput_values = pd.DataFrame.from_records(throughput_values, columns=['time', 'value'])
            throughput_values['value'] = throughput_values['value'].astype(float)
            # groupby time and measure mean: it happens when data is for all service ves=rsions
            throughput_values = throughput_values.groupby('time').mean().reset_index()


            ################################################
            #Put all metrics in a dictionary

           # metrics= pd.merge(request_duration_values, request_byte_values, response_byte_values,queue_size_values,latency_values,throughput_values,on="movie_title")
            df_dict = {
                'request_duration_values': request_duration_values,
                'request_byte_values': request_byte_values,
                'response_byte_values': response_byte_values,
                'queue_size_values': queue_size_values,
                'latency_values': latency_values,
                'throughput_values': throughput_values,
            }

            #set time to datetime type
            for _, df in df_dict.items():
                # df['time'] = pd.to_datetime(df['time'])
                df['time'] = df['time'].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y - %m - %d  %H : %M : %S"))

            #set time as index
            for _, df in df_dict.items():
                df.set_index('time', inplace=True)

            #concat all metrics df based on time col
            metrics = pd.concat([request_duration_values, request_byte_values, response_byte_values,queue_size_values,
                                 latency_values,throughput_values], axis=1, sort=False).reset_index()
            # remove null
            metrics= metrics.dropna()

           #remove time col
            metrics = metrics.drop(columns='time')
            #get the col names
            metrics.columns=['request_duration', 'request_byte', 'response_byte', 'queue_size',
                    'latency', 'throughput']
            metrics=metrics.astype('double')
            #reset index col
            metrics.reset_index(drop=True, inplace=True)

            #print(metrics['queue_size'].loc[metrics['queue_size'] != 0])
            #print(metrics.loc[metrics['latency'] != 0])
            #print(metrics)

            #normalize by mean
            #metrics = (metrics - metrics.mean()) / metrics.std()
            print("The length of the colllected metric is:"+ str(len(metrics)) )
            return metrics
    #avg number of relationships per edge
    def Num_Relationships(self,rel):
        num_rel=0
        for edge_relation_ls in rel:
            #
            for dic in edge_relation_ls:
                #count numbers of relationship
                num_rel+=len(dic["relations"])
        num_edges= len(rel)
        avg=num_rel/num_edges
        return avg
























