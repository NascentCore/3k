from locust import HttpUser, task

class HelloWorldUser(HttpUser):
    @task
    def hello_world(self):
        self.client.post(url="/v1/chat/completions",json={"model":"/data2/dg/models/meta-llama-3.1-8b-instruct","messages":[{"role":"user","content":"hello world"}]})