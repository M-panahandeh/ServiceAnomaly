import json
from typing import Dict, List
import json
import tkinter as tk
from tkinter import filedialog


class Span:
    def __init__(self, spanID: str, operationName: str, startTime: str, duration: int, tags: List[Dict[str, str]],
                 references: List[Dict[str, str]]):
        self.spanID = spanID
        self.operationName = operationName
        self.startTime = startTime
        self.duration = duration
        self.tags = tags
        self.references = references
        self.children = []

    def add_child(self, child):
        self.children.append(child)


def parse_spans(spans):
    parsed_spans = {}
    for span in spans:
        parsed_spans[span["spanID"]] = Span(span["spanID"], span["operationName"], span["startTime"], span["duration"],
                                            span["tags"], span["references"])
    return parsed_spans


def build_dag(spans):
    parsed_spans = parse_spans(spans)
    for span in parsed_spans.values():
        for reference in span.references:
            if reference["spanID"] in parsed_spans:
                parent = parsed_spans[reference["spanID"]]
                parent.add_child(span)
            else:
                parent = None  # set parent to None if parsed_spans[reference["spanID"]] doesn't exist
    return parsed_spans


def generate_dag(trace):
    traceID = trace["traceID"]
    spans = trace["spans"]
    parsed_spans = build_dag(spans)
    print(f"Trace ID: {traceID}")
    for span in parsed_spans.values():
        print(
            f"Span ID: {span.spanID} Operation: {span.operationName} Start Time: {span.startTime} Duration: {span.duration}")
        for child in span.children:
            print(f"    Child: {child.spanID} Operation: {child.operationName}")


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
generate_dag(trace)

