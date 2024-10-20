"""test-transformers: A Flower / HuggingFace app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from transformers import AutoModelForSequenceClassification

from test_transformers.task import get_weights, load_data, set_weights, test, train
from random import randint

# Flower client
class FlowerClient(NumPyClient):
    def __init__(self, net, partition_id, trainloader, testloader, local_epochs, client_metrics):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.partition_id = partition_id
        self.client_metrics = client_metrics

    def fit(self, parameters, config):
        print(f"client {self.partition_id}: start training with client_metrics {self.client_metrics}")
        set_weights(self.net, parameters)
        train(self.net, self.trainloader, epochs=self.local_epochs, device=self.device)
        return get_weights(self.net), len(self.trainloader), self.client_metrics

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return float(loss), len(self.testloader), {"accuracy": accuracy}


def client_fn(context: Context, config: dict):

    # Get this client's dataset partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    model_name = config["model"]
    dataset = config["dataset"]
    trainloader, valloader = load_data(partition_id, num_partitions, model_name, dataset)

    # Load model
    num_labels = 2
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    local_epochs = 1
    clients = config["clients"]
    client_metrics = clients[randint(0,len(clients)-1)]
    # Return Client instance
    return FlowerClient(net, partition_id,trainloader, valloader, local_epochs, client_metrics).to_client()

def create_client_app(config):
    return ClientApp(client_fn=lambda context: client_fn(context, config))