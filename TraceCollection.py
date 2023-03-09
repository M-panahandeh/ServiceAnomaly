import requests
import json
    # class definition
class TraceCollection:

        data = []

        def Trace2json(self):
            url = "http://localhost:16686/jaeger/api/dependencies?endTs=1639432124762&lookback=864000000"
            resp = requests.get(url)
            # print(resp.status_code)
            # print(resp.content)
            self.data = resp.content
            self.data = self.data.decode('utf-8')  # string
            # write to Josn
            with open('./traces.json', 'w') as outfile:
                json.dump(self.data, outfile)