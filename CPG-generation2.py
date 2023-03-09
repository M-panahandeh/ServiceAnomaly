
##based on processID
from typing import Dict, List
import json
import tkinter as tk
from tkinter import filedialog


import json

def create_dag(trace):
    # Create a dictionary to store spans by ID
    span_map = {}
    for span in trace['spans']:
        span_id = span['spanID']
        span_map[span_id] = span

    # Create a list of edges in the DAG
    dag = []
    for span in trace['spans']:
        span_id = span['spanID']
        parent_id = None
        if span['references']:
            ref_type = span['references'][0]['refType']
            if ref_type == 'CHILD_OF':
                parent_id = span['references'][0]['spanID']
            elif ref_type == 'FOLLOWS_FROM':
                parent_id = span_map[span['references'][0]['spanID']]['parentSpanID']
        else:
            parent_id = 'null'

        if parent_id != None and parent_id != 'null':
            if (parent_id == trace['traceID']):
                child_process_id = span['spanID']
                dag.append(('root', child_process_id))

            else:
                parent_process_id = span_map[parent_id]['spanID']
                child_process_id = span['spanID']
                dag.append((parent_process_id, child_process_id))

        else:
            dag.append(('null', span['spanID']))
    #return dag
    #return dag : if you wnat day of spanIDs
    #check if we have multiple traceId=spanID that is when we have multiple (root,X). if X-spanID has a child it is fine.otherwise it is a called
    # unit of work with no parnt (null,X) or it is a caller calling a null (x,null). it can be both according to the getting data from source or destination.
    #we use the second one (getting data from destination view). Therefore, we use process_id which is the called piece (destination). and we change the edge (root,x) to (x-processid , null)
    for edge in dag:
        if edge[0] == 'root': #check to see if we have multiple root check to see if spanID is parent of something.
            has_child = False
            for other_edge in dag:
                if other_edge[0] == edge[1]:
                    has_child = True
                    break
            if has_child:
                continue
            else:
                dag[dag.index(edge)] = (edge[1], None)
    #use processIDs
    for i, edge in enumerate(dag):
        new_edge = []
        for element in edge:
            if element not in (None, "root", "null"):
                new_edge.append(span_map[element]['processID'])
            else:
                new_edge.append(element)
        dag[i] = tuple(new_edge)
    # Return the DAG as a list of edges
    return dag

# Create a Tkinter root window (optional)
root = tk.Tk()
root.withdraw()

# Ask the user to select a JSON file using a file dialog
file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])

    # Read the contents of the file into a string variable
with open(file_path, 'r') as f:
    json_str = f.read()
trace = json.loads(json_str)
trace = trace['data'][0]
# trace= '''
#         {
#             "traceID": "868cc4680a73456067e12021f1451995",
#             "spans": [
#                 {
#                     "traceID": "868cc4680a73456067e12021f1451995",
#                     "spanID": "67e12021f1451995",
#                     "operationName": "teastore-registry.default.svc.cluster.local:8080/*",
#                     "references": [],
#                     "startTime": 1659572858861200,
#                     "duration": 5003897,
#                     "tags": [
#                         {
#                             "key": "http.method",
#                             "type": "string",
#                             "value": "GET"
#                         },
#                         {
#                             "key": "istio.mesh_id",
#                             "type": "string",
#                             "value": "cluster.local"
#                         },
#                         {
#                             "key": "response_flags",
#                             "type": "string",
#                             "value": "DI,DC"
#                         },
#                         {
#                             "key": "istio.canonical_revision",
#                             "type": "string",
#                             "value": "latest"
#                         },
#                         {
#                             "key": "peer.address",
#                             "type": "string",
#                             "value": "172.17.0.20"
#                         },
#                         {
#                             "key": "error",
#                             "type": "bool",
#                             "value": true
#                         },
#                         {
#                             "key": "request_size",
#                             "type": "string",
#                             "value": "0"
#                         },
#                         {
#                             "key": "guid:x-request-id",
#                             "type": "string",
#                             "value": "a65a3692-e1d0-911c-843b-805d304636d1"
#                         },
#                         {
#                             "key": "response_size",
#                             "type": "string",
#                             "value": "0"
#                         },
#                         {
#                             "key": "node_id",
#                             "type": "string",
#                             "value": "sidecar~172.17.0.20~teastore-auth-84f7654d8d-72wlt.default~default.svc.cluster.local"
#                         },
#                         {
#                             "key": "http.url",
#                             "type": "string",
#                             "value": "http://teastore-registry:8080/tools.descartes.teastore.registry/rest/services/tools.descartes.teastore.persistence/"
#                         },
#                         {
#                             "key": "downstream_cluster",
#                             "type": "string",
#                             "value": "-"
#                         },
#                         {
#                             "key": "istio.namespace",
#                             "type": "string",
#                             "value": "default"
#                         },
#                         {
#                             "key": "component",
#                             "type": "string",
#                             "value": "proxy"
#                         },
#                         {
#                             "key": "istio.canonical_service",
#                             "type": "string",
#                             "value": "teastore"
#                         },
#                         {
#                             "key": "http.protocol",
#                             "type": "string",
#                             "value": "HTTP/1.1"
#                         },
#                         {
#                             "key": "http.status_code",
#                             "type": "string",
#                             "value": "0"
#                         },
#                         {
#                             "key": "user_agent",
#                             "type": "string",
#                             "value": "Jersey/3.0.2 (HttpUrlConnection 11.0.11)"
#                         },
#                         {
#                             "key": "span.kind",
#                             "type": "string",
#                             "value": "client"
#                         },
#                         {
#                             "key": "internal.span.format",
#                             "type": "string",
#                             "value": "zipkin"
#                         }
#                     ],
#                     "logs": [],
#                     "processID": "p1",
#                     "warnings": null
#                 }
#             ],
#             "processes": {
#                 "p1": {
#                     "serviceName": "teastore.default",
#                     "tags": [
#                         {
#                             "key": "ip",
#                             "type": "string",
#                             "value": "172.17.0.20"
#                         }
#                     ]
#                 }
#             },
#             "warnings": null
#         }
#     '''
# trace = json.loads(trace)
dag = create_dag(trace)
print(dag)
