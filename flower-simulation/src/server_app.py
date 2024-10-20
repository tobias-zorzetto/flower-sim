"""test-transformers: A Flower / HuggingFace app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from transformers import AutoModelForSequenceClassification

from test_transformers.task import get_weights

class CustomStrategy(FedAvg):
    def __init__(self, fraction_fit=1.0, fraction_evaluate=1.0,initial_parameters=None):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            initial_parameters=initial_parameters,
        )
        self.cumulative_metrics = {}

    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights, _ = super().aggregate_fit(rnd, results, failures)
        for _, client_result in results:
            for metric_name, metric_value in client_result.metrics.items():
                if metric_name in self.cumulative_metrics:
                    self.cumulative_metrics[metric_name] += metric_value
                else:
                    self.cumulative_metrics[metric_name] = metric_value
        print(f"After round {rnd}, cumulative metrics: {self.cumulative_metrics}")
        
        return aggregated_weights, {}

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_loss, _ = super().aggregate_evaluate(rnd, results, failures)
        for _, client_result in results:
            for metric_name, metric_value in client_result.metrics.items():
                if metric_name in self.cumulative_metrics:
                    self.cumulative_metrics[metric_name] += metric_value
                else:
                    self.cumulative_metrics[metric_name] = metric_value
        print(f"After round {rnd}, cumulative evaluation metrics: {self.cumulative_metrics}")
        
        return aggregated_loss, {}


def server_fn(context: Context, config):
    num_rounds = 3
    fraction_fit = 0.5

    model_name = config["model"]
    num_labels = 2
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    weights = get_weights(net)
    initial_parameters = ndarrays_to_parameters(weights)

    # Define strategy
    strategy = CustomStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

def create_server_app(config):
    return ServerApp(server_fn=lambda context: server_fn(context, config))